from dataclasses import dataclass

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class LSTMModule:
    vocab: int
    embed: int
    hidden: int
    layers: int
    dropout: float
    pad_id: int


class LuongAttention(torch.nn.Module):
    def forward(
        self, enc_mask: torch.Tensor, enc_out: torch.Tensor, dec_hid: torch.Tensor
    ) -> torch.Tensor:
        sims = torch.bmm(enc_out, dec_hid.unsqueeze(2)).squeeze(2)
        sims = sims.masked_fill(enc_mask, float("-inf"))
        attention = torch.softmax(sims, dim=1)
        context = torch.bmm(attention.unsqueeze(1), enc_out)
        return context.squeeze(1)


class EncoderProjection(torch.nn.Module):
    def __init__(self, enc_size: int, dec_size: int, layers: int) -> None:
        super().__init__()
        self.proj_h = torch.nn.ModuleList(
            torch.nn.Linear(2 * enc_size, dec_size) for _ in range(layers)
        )
        self.proj_c = torch.nn.ModuleList(
            torch.nn.Linear(2 * enc_size, dec_size) for _ in range(layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.stack([torch.tanh(layer(x)) for layer in self.proj_h], dim=0)
        c0 = torch.stack([torch.tanh(layer(x)) for layer in self.proj_c], dim=0)
        return h0, c0


class G2PLSTM(torch.nn.Module):
    def __init__(self, enc: LSTMModule, dec: LSTMModule) -> None:
        super().__init__()
        self.enc, self.dec = enc, dec
        self.enc_embeds = torch.nn.Embedding(
            enc.vocab, enc.embed, padding_idx=enc.pad_id
        )
        self.enc_lstm = torch.nn.LSTM(
            enc.embed,
            enc.hidden,
            enc.layers,
            dropout=enc.dropout,
            batch_first=True,
            bidirectional=True,
        )

        self.dec_embeds = torch.nn.Embedding(
            dec.vocab, dec.embed, padding_idx=dec.pad_id
        )
        self.dec_lstm = torch.nn.LSTM(
            dec.embed, dec.hidden, dec.layers, dropout=dec.dropout, batch_first=True
        )

        self.fc = torch.nn.Linear(2 * dec.hidden, dec.vocab)
        self.proj = torch.nn.Linear(2 * enc.hidden, dec.hidden)
        self.enc_proj = EncoderProjection(enc.hidden, dec.hidden, dec.layers)

        self.luong_attn = LuongAttention()

    def forward(
        self,
        enc_in: torch.Tensor,
        enc_lens: torch.Tensor,
        dec_in: torch.Tensor,
        teacher_p: float,
    ) -> torch.Tensor:
        assert teacher_p <= 1.0
        B = enc_in.size(0)
        enc_embed = self.enc_embeds(enc_in)
        packed_enc = torch.nn.utils.rnn.pack_padded_sequence(
            enc_embed, enc_lens.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, (h, c) = self.enc_lstm(packed_enc)
        enc_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )

        enc_out = self.proj(enc_out)
        enc_mask = enc_in == self.enc.pad_id

        init_h, init_c = self.enc_proj(torch.cat([h[-2], h[-1]], dim=1))
        dec_hidden, dec_cell = init_h, init_c

        outputs = []
        t_logits = dec_in[:, 0]
        for t in range(dec_in.size(1)):
            if not t:
                dec_in_t = dec_in[:, t]
            else:
                use_teacher = torch.rand(B, device=device) < teacher_p
                dec_in_t = torch.where(
                    use_teacher, dec_in[:, t], t_logits.argmax(dim=1)
                )

            dec_embed_t = self.dec_embeds(dec_in_t).unsqueeze(1)
            dec_out, (dec_hidden, dec_cell) = self.dec_lstm(
                dec_embed_t, (dec_hidden, dec_cell)
            )

            context = self.luong_attn(enc_mask, enc_out, dec_hidden[-1])
            cat = torch.cat([dec_out.squeeze(1), context], dim=1)

            t_logits = self.fc(cat)
            outputs.append(t_logits.unsqueeze(1))

        logits = torch.cat(outputs, dim=1)
        return logits
