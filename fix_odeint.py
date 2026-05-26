import re

with open('grim_nlp_experiment.py', 'r', encoding='utf-8') as f:
    content = f.read()

# odeint_adjoint の呼び出しを置換（adjoint_params を明示）
old_call = r'''solution = odeint_adjoint\(
            self\.vector_field,
            psi_0,
            t_span,
            method='dopri5',
            rtol=1e-4,
            atol=1e-6
        \)'''

new_call = '''solution = odeint_adjoint(
            self.vector_field,
            psi_0,
            t_span,
            method='dopri5',
            rtol=1e-4,
            atol=1e-6,
            adjoint_params=()
        )'''

content = re.sub(old_call, new_call, content, flags=re.DOTALL)

with open('grim_nlp_experiment.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed odeint_adjoint call")
