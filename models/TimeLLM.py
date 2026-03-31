from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        _eager = getattr(configs, 'use_eager_attention', False)

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = False
            self.llama_config.output_hidden_states = False
            if _eager:
                self.llama_config._attn_implementation = 'eager'
            _llama_hf_id = 'huggyllama/llama-7b'
            _local_load_errors = (OSError, EnvironmentError, AttributeError, ValueError)
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    _llama_hf_id,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                )
            except _local_load_errors as e:
                print(f"Local Llama load failed ({type(e).__name__}: {e}). Fetching from Hugging Face...")
                self.llm_model = LlamaModel.from_pretrained(
                    _llama_hf_id,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    _llama_hf_id,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except _local_load_errors as e:
                print(f"Local tokenizer load failed ({type(e).__name__}: {e}). Fetching from Hugging Face...")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    _llama_hf_id,
                    trust_remote_code=True,
                    local_files_only=False,
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = False
            self.gpt2_config.output_hidden_states = False
            if _eager:
                self.gpt2_config._attn_implementation = 'eager'
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = False
            self.bert_config.output_hidden_states = False
            if _eager:
                self.bert_config._attn_implementation = 'eager'
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = getattr(configs, 'num_tokens', 1000)
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                return_aux=False):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            result = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                   return_aux=return_aux)
            if return_aux:
                aux = result
                aux['pred'] = aux['pred'][:, -self.pred_len:, :]
                return aux
            return result[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_aux=False):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))

        if return_aux:
            enc_out, reprog_attn = self.reprogramming_layer(
                enc_out, source_embeddings, source_embeddings,
                return_attention=True)
        else:
            enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        reprog_out = enc_out  # [B*N, patch_nums, d_llm]

        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        prompt_len = prompt_embeddings.shape[1]

        if return_aux:
            prev_hs = self.llm_model.config.output_hidden_states
            self.llm_model.config.output_hidden_states = True
            _has_attns = False
            try:
                prev_att = self.llm_model.config.output_attentions
                self.llm_model.config.output_attentions = True
                _has_attns = True
            except (ValueError, AttributeError):
                pass

        llm_out = self.llm_model(inputs_embeds=llama_enc_out)

        if return_aux:
            self.llm_model.config.output_hidden_states = prev_hs
            if _has_attns:
                self.llm_model.config.output_attentions = prev_att

        dec_out = llm_out.last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        pre_head = dec_out[:, :, :, -self.patch_nums:]  # [B, n_vars, d_ff, patch_nums]

        dec_out = self.output_projection(pre_head)
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        if not return_aux:
            return dec_out

        # --- build auxiliary dict ---
        # hidden_states: tuple of (n_layers+1) tensors [B*N, seq_total, d_llm]
        # reshape each to [B, n_vars, seq_total, d_llm], slice to patch span,
        # and truncate feature dim to d_ff for alignment with the head.
        h_layers = []
        for h in llm_out.hidden_states:
            hl = h[:, :, :self.d_ff]                          # [B*N, seq_total, d_ff]
            hl = hl.reshape(-1, n_vars, hl.shape[1], hl.shape[2])
            hl = hl.permute(0, 1, 3, 2)                      # [B, n_vars, d_ff, seq_total]
            hl = hl[:, :, :, -self.patch_nums:]               # [B, n_vars, d_ff, patch_nums]
            h_layers.append(hl.detach().cpu())

        llm_attns = []
        if llm_out.attentions is not None:
            for attn in llm_out.attentions:
                llm_attns.append(attn.detach().cpu())

        reprog_out_shaped = reprog_out.reshape(
            -1, n_vars, reprog_out.shape[1], reprog_out.shape[2])  # [B, n_vars, patch_nums, d_llm]

        return {
            'pred': dec_out,
            'h_layers': h_layers,
            'h_last': h_layers[-1],
            'pre_head': pre_head.detach().cpu(),        # [B, n_vars, d_ff, patch_nums]
            'reprog_out': reprog_out_shaped.detach().cpu(),
            'reprog_attn': reprog_attn.detach().cpu(),  # [B*N, n_heads, patch_nums, S]
            'llm_attns': llm_attns,
            'prompt_len': prompt_len,
            'n_vars': n_vars,
        }

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding,
                return_attention=False):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out, A = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)
        out = self.out_projection(out)

        if return_attention:
            return out, A
        return out

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding, A
