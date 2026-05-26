import re

with open('grim_nlp_experiment.py', 'r', encoding='utf-8') as f:
    content = f.read()

# ode_step メソッドを置換
old_method = r'''    def ode_step\(self, psi_0: torch\.Tensor\) -> Tuple\[torch\.Tensor, int\]:
        """
        ODE 積分：dψ/dt = v_⊥\(ψ\), t∈\[0,1\]
        returns: \(psi_T, num_steps\)
        """
        batch_size = psi_0\.shape\[0\]
        t_span = torch\.tensor\(\[0\.0, 1\.0\], device=psi_0\.device\)

        # ODE ソルバー（DOPRI5）
        # torchdiffeq の odeint_adjoint を使用
        solution = odeint_adjoint\(
            lambda t, psi: self\.vector_field\(t, psi, psi_0\),
            psi_0,
            t_span,
            method='dopri5',
            rtol=1e-4,
            atol=1e-6
        \)'''

new_method = '''    def ode_step(self, psi_0: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        ODE 積分：dψ/dt = v_⊥(ψ), t∈[0,1]
        returns: (psi_T, num_steps)
        """
        batch_size = psi_0.shape[0]
        t_span = torch.tensor([0.0, 1.0], device=psi_0.device)

        # VectorField に psi_0 を設定
        self.vector_field.set_psi_0(psi_0)

        # ODE ソルバー（DOPRI5）
        # torchdiffeq の odeint_adjoint を使用
        solution = odeint_adjoint(
            self.vector_field,
            psi_0,
            t_span,
            method='dopri5',
            rtol=1e-4,
            atol=1e-6
        )'''

content = re.sub(old_method, new_method, content, flags=re.DOTALL)

with open('grim_nlp_experiment.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed ode_step method")
