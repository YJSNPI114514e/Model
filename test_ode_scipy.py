"""ODE ソルバーの scipy 移行テストと検証"""

import torch
import numpy as np
from scipy.integrate import solve_ivp
from torchdiffeq import odeint

def test_ode_scipy_migration():
    """torchdiffeq から scipy.integrate.solve_ivp への移行テスト"""
    
    # テスト用微分方程式: dψ/dt = -i * H * ψ (単純な調和振動子)
    def dynamics_torch(t, y_flat):
        D = 4
        psi = torch.view_as_complex(y_flat.view(D, 2))  # [D]
        H = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.complex64))
        v = -1j * H.mv(psi)  # [D]
        return torch.view_as_real(v).reshape(-1)
    
    def dynamics_scipy(t, y_flat):
        psi = y_flat[:4] + 1j * y_flat[4:]
        H = np.diag([1.0, 2.0, 3.0, 4.0])
        v = -1j * H @ psi
        return np.concatenate([v.real, v.imag])
    
    # 初期状態
    psi0 = torch.randn(4, dtype=torch.complex64)
    psi0 = psi0 / torch.norm(psi0)
    y0_torch = torch.view_as_real(psi0).reshape(-1).numpy().copy()
    
    # torchdiffeq で積分
    t_span = (0.0, 1.0)
    t_eval = torch.tensor([t_span[0], t_span[1]], dtype=torch.float64)
    
    solution_torch = odeint(dynamics_torch, torch.from_numpy(y0_torch).float(), 
                           t_eval, method='dopri5', rtol=1e-4, atol=1e-6)
    psi_T_torch = torch.view_as_complex(solution_torch[-1].view(4, 2))
    
    # scipy で積分
    sol_scipy = solve_ivp(dynamics_scipy, t_span, y0_torch, 
                         method='RK45', rtol=1e-4, atol=1e-6, dense_output=False)
    psi_T_scipy_np = sol_scipy.y[:, -1]
    psi_T_scipy = torch.from_numpy(psi_T_scipy_np[:4] + 1j * psi_T_scipy_np[4:])
    
    # 比較
    print("=== ODE ソルバー比較 ===")
    print(f"torchdiffeq 結果：{psi_T_torch}")
    print(f"scipy 結果：{psi_T_scipy}")
    
    # ノルムチェック
    norm_torch = torch.norm(psi_T_torch).item()
    norm_scipy = torch.norm(psi_T_scipy).item()
    print(f"\nノルム (torchdiffeq): {norm_torch:.8f} (理想：1.0)")
    print(f"ノルム (scipy): {norm_scipy:.8f} (理想：1.0)")
    
    # 相対誤差
    rel_error = torch.abs(psi_T_torch - psi_T_scipy).max().item()
    print(f"\n最大相対誤差：{rel_error:.2e}")
    print(f"許容誤差 (1e-3): {'PASS' if rel_error < 1e-3 else 'FAIL'}")
    
    return rel_error < 1e-3

if __name__ == "__main__":
    success = test_ode_scipy_migration()
    print(f"\nテスト結果：{'SUCCESS' if success else 'FAILURE'}")
