import re

with open('grim_nlp_experiment.py', 'r', encoding='utf-8') as f:
    content = f.read()

# VectorField クラスを置換
old_class = r'''class VectorField\(nn\.Module\):
    """ベクトル場 v_⊥.*?return v_perp'''

new_class = '''class VectorField(nn.Module):
    """ベクトル場 v_⊥(ψ) = v(ψ) - ⟨ψ|v(ψ)⟩|ψ⟩（接空間射影付き）"""
    
    def __init__(self):
        super().__init__()
        self.psi_0_buf = None
        
    def set_psi_0(self, psi_0: torch.Tensor):
        """psi_0 を設定"""
        self.psi_0_buf = psi_0
        
    def forward(self, t: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """
        t: 時間（未使用だが odeint のために必要）
        psi: [batch, D] 現在の状態
        returns: dψ/dt [batch, D]
        """
        psi_0 = self.psi_0_buf
        
        # v(ψ) = (⟨ψ_0|ψ⟩ / (|⟨ψ|ψ_0⟩|^2+ε)) |ψ_0⟩
        inner = torch.sum(psi_0.conj() * psi, dim=-1, keepdim=True)  # [batch, 1]
        denom = torch.abs(inner)**2 + EPS
        v = (inner / denom) * psi_0  # [batch, D]
        
        # 接空間射影：v_⊥ = v - ⟨ψ|v⟩ψ
        proj_coeff = torch.sum(psi.conj() * v, dim=-1, keepdim=True)  # [batch, 1]
        v_perp = v - proj_coeff * psi
        
        return v_perp'''

content = re.sub(old_class, new_class, content, flags=re.DOTALL)

with open('grim_nlp_experiment.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed VectorField class")
