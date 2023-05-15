from set_transformer_modules import *

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False,
            num_sabs=2, dropout=0.0):
        super(SetTransformer, self).__init__()

        if num_sabs == 2:
            self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln, dropout=dropout),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln, dropout=dropout),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            )
            
        elif num_sabs == 3:
            self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln, dropout=dropout),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln, dropout=dropout),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln, dropout=dropout),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            )

        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln, dropout=dropout),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X)) 
