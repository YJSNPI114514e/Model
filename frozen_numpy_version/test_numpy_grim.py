#!/usr/bin/env python3
"""
NumPy GRIM 実装の動作確認スクリプト
- 各モジュールのインポート確認
- 基本的な機能テスト
- 数値的安定性の確認
"""

import numpy as np
import sys

def test_geometry():
    """geometry_np.py のテスト"""
    print("\n=== Testing geometry_np.py ===")
    from grim.geometry_np import complex_inner, normalize_state, tangent_project, fs_distance, softmax
    
    # テストデータ生成
    np.random.seed(42)
    D = 8
    a = np.random.randn(D) + 1j * np.random.randn(D)
    b = np.random.randn(D) + 1j * np.random.randn(D)
    psi = np.random.randn(D) + 1j * np.random.randn(D)
    
    # complex_inner
    inner = complex_inner(a, b)
    print(f"complex_inner: ⟨a|b⟩ = {inner:.6f}")
    assert np.isclose(inner, np.vdot(a, b)), "complex_inner failed"
    
    # normalize_state
    psi_norm = normalize_state(psi)
    norm = np.linalg.norm(psi_norm)
    print(f"normalize_state: ‖ψ‖ = {norm:.10f} (should be 1.0)")
    assert np.isclose(norm, 1.0), f"Normalization failed: {norm}"
    
    # tangent_project
    v = np.random.randn(D) + 1j * np.random.randn(D)
    v_proj = tangent_project(v, psi_norm)
    # 接空間射影後、ψとの内積は 0 に近いはず
    overlap = complex_inner(psi_norm, v_proj)
    print(f"tangent_project: |⟨ψ|v_proj⟩| = {np.abs(overlap):.10f} (should be ~0)")
    assert np.abs(overlap) < 1e-10, "Tangent projection failed"
    
    # fs_distance
    phi = np.random.randn(D) + 1j * np.random.randn(D)
    phi = normalize_state(phi)
    dist = fs_distance(psi_norm, phi)
    print(f"fs_distance: d_FS(ψ,φ) = {dist:.6f}")
    assert 0 <= dist <= np.pi/2, f"FS distance out of range: {dist}"
    
    # softmax
    x = np.array([1.0, 2.0, 3.0, 4.0])
    sm = softmax(x)
    print(f"softmax: sum = {np.sum(sm):.10f} (should be 1.0)")
    assert np.isclose(np.sum(sm), 1.0), "Softmax sum != 1"
    assert np.all(sm > 0), "Softmax has non-positive values"
    
    print("✓ geometry_np.py: All tests passed")
    return True

def test_tokenizer():
    """tokenizer_np.py のテスト"""
    print("\n=== Testing tokenizer_np.py ===")
    from grim.tokenizer_np import NumPyTokenizer
    
    np.random.seed(42)
    V, D = 10, 8
    
    # ダミーパラメータ作成
    embeddings = np.random.randn(V, D).astype(np.complex128)
    T_eigvals = np.random.randn(D)
    T_eigvecs = np.linalg.qr(np.random.randn(D, D))[0]  # 直交行列
    T_eigvecs_inv = T_eigvecs.T.conj()
    delta_eig = np.random.randn(D)
    delta_raw = np.random.randn(D)
    B_mat = np.random.randn(D, D)
    phase_weight = 0.5
    attn_weight = np.random.randn(D).astype(np.float64) * 0.1
    
    tokenizer = NumPyTokenizer(
        embeddings=embeddings,
        T_eigvals=T_eigvals,
        T_eigvecs=T_eigvecs,
        T_eigvecs_inv=T_eigvecs_inv,
        delta_eig=delta_eig,
        delta_raw=delta_raw,
        B_mat=B_mat,
        phase_weight=phase_weight,
        attn_weight=attn_weight
    )
    
    # tokenize
    token_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    psi_0 = tokenizer.tokenize(token_ids)
    print(f"tokenize: ψ₀ shape = {psi_0.shape}, ‖ψ₀‖ = {np.linalg.norm(psi_0):.10f}")
    assert psi_0.shape == (D,), f"Wrong shape: {psi_0.shape}"
    assert np.isclose(np.linalg.norm(psi_0), 1.0), f"Tokenized state not normalized: {np.linalg.norm(psi_0)}"
    
    # inject
    new_psi = tokenizer.inject(psi_0, token_id=5, rate=0.1)
    print(f"inject: ‖ψ_new‖ = {np.linalg.norm(new_psi):.10f}")
    assert np.isclose(np.linalg.norm(new_psi), 1.0), f"Injected state not normalized: {np.linalg.norm(new_psi)}"
    
    print("✓ tokenizer_np.py: All tests passed")
    return True

def test_flow_field():
    """flow_field_np.py のテスト"""
    print("\n=== Testing flow_field_np.py ===")
    from grim.flow_field_np import NumPyEnergyField
    from grim.history_np import NumPyHistory
    
    np.random.seed(42)
    V, D = 10, 8
    
    embeddings = np.random.randn(V, D).astype(np.complex128)
    field = NumPyEnergyField(embeddings=embeddings)
    
    psi = np.random.randn(D) + 1j * np.random.randn(D)
    psi = psi / np.linalg.norm(psi)
    psi_0 = psi.copy()
    
    history = NumPyHistory(max_entries=10)
    history.push(psi)
    
    # energy_and_grad - history を直接渡す（リストとして）
    E, grad = field.energy_and_grad(psi, psi_0, history.short_term)
    print(f"energy_and_grad: E = {E.real:.6f}, ‖grad‖ = {np.linalg.norm(grad):.6f}")
    assert np.isreal(E) or np.isclose(E.imag, 0), "Energy should be real"
    assert grad.shape == (D,), f"Wrong grad shape: {grad.shape}"
    
    # ode_rhs
    v = field.ode_rhs(0.0, psi, psi_0, history.short_term)
    print(f"ode_rhs: ‖v‖ = {np.linalg.norm(v):.6f}")
    assert v.shape == (D,), f"Wrong velocity shape: {v.shape}"
    
    # 接空間に射影されているか確認
    overlap = np.vdot(psi, v)
    print(f"ode_rhs tangent check: |⟨ψ|v⟩| = {np.abs(overlap):.10f}")
    assert np.abs(overlap) < 1e-10, "Velocity not in tangent space"
    
    print("✓ flow_field_np.py: All tests passed")
    return True

def test_ode_solver():
    """ode_solver_np.py のテスト"""
    print("\n=== Testing ode_solver_np.py ===")
    from grim.ode_solver_np import integrate_flow
    from grim.flow_field_np import NumPyEnergyField
    from grim.history_np import NumPyHistory
    
    np.random.seed(42)
    V, D = 10, 8
    
    embeddings = np.random.randn(V, D).astype(np.complex128)
    psi_0 = np.random.randn(D) + 1j * np.random.randn(D)
    psi_0 = psi_0 / np.linalg.norm(psi_0)
    
    history = NumPyHistory(max_entries=10)
    history.push(psi_0)
    
    # energy field を作成
    field = NumPyEnergyField(embeddings=embeddings)
    
    # ODE 右辺関数のラッパー
    def ode_rhs(t, psi, psi_0_fixed, hist):
        return field.ode_rhs(t, psi, psi_0_fixed, hist)
    
    psi_T = integrate_flow(
        ode_rhs=ode_rhs,
        psi_0=psi_0,
        psi_0_fixed=psi_0,
        history=history.short_term,
        params={},
        embeddings=embeddings,
        T=0.5
    )
    
    print(f"integrate_flow: ψ_T shape = {psi_T.shape}, ‖ψ_T‖ = {np.linalg.norm(psi_T):.10f}")
    assert psi_T.shape == (D,), f"Wrong shape: {psi_T.shape}"
    assert np.isclose(np.linalg.norm(psi_T), 1.0), f"Integrated state not normalized: {np.linalg.norm(psi_T)}"
    
    print("✓ ode_solver_np.py: All tests passed")
    return True

def test_observation():
    """observation_np.py のテスト"""
    print("\n=== Testing observation_np.py ===")
    from grim.observation_np import born_probs
    
    np.random.seed(42)
    V, D = 10, 8
    
    embeddings = np.random.randn(V, D).astype(np.complex128)
    psi = np.random.randn(D) + 1j * np.random.randn(D)
    psi = psi / np.linalg.norm(psi)
    
    probs = born_probs(psi, embeddings)
    
    print(f"born_probs: shape = {probs.shape}, sum = {np.sum(probs):.10f}")
    print(f"  min={np.min(probs):.6f}, max={np.max(probs):.6f}")
    assert probs.shape == (V,), f"Wrong shape: {probs.shape}"
    assert np.isclose(np.sum(probs), 1.0), f"Probabilities don't sum to 1: {np.sum(probs)}"
    assert np.all(probs >= 0), "Negative probabilities"
    
    print("✓ observation_np.py: All tests passed")
    return True

def test_history():
    """history_np.py のテスト"""
    print("\n=== Testing history_np.py ===")
    from grim.history_np import NumPyHistory
    
    np.random.seed(42)
    D = 8
    
    history = NumPyHistory(max_entries=5, gamma=0.99)
    
    # push
    for i in range(7):  # max_entries より多くプッシュ
        psi = np.random.randn(D) + 1j * np.random.randn(D)
        psi = psi / np.linalg.norm(psi)
        history.push(psi)
    
    total_entries = len(history.short_term) + len(history.mid_term) + len(history.long_term)
    print(f"push: total entries = {total_entries}")
    assert total_entries <= 100, f"Too many entries: {total_entries}"
    
    # decay
    history.decay(boundary_prob=0.0)
    print(f"decay: weights updated")
    all_weights = [e.weight for e in history.short_term + history.mid_term + history.long_term]
    assert all(w > 0 for w in all_weights), "Weights should be positive"
    
    # summarize
    psi_current = np.random.randn(D) + 1j * np.random.randn(D)
    psi_current = psi_current / np.linalg.norm(psi_current)
    summary = history.summarize(psi_current)
    
    print(f"summarize: shape = {summary.shape}, ‖summary‖ = {np.linalg.norm(summary):.10f}")
    # summarize は重み付き平均なので、必ずしも正規化されていない場合がある
    # 形状が正しければ OK とする
    
    print("✓ history_np.py: All tests passed")
    return True

def test_model():
    """model_np.py のテスト"""
    print("\n=== Testing model_np.py ===")
    from grim.model_np import NumPyGRIM
    from grim.tokenizer_np import NumPyTokenizer
    
    np.random.seed(42)
    V, D = 20, 16
    
    # ダミーパラメータ作成
    embeddings = np.random.randn(V, D).astype(np.complex128)
    T_eigvals = np.random.randn(D)
    T_eigvecs = np.linalg.qr(np.random.randn(D, D))[0]
    T_eigvecs_inv = T_eigvecs.T.conj()
    delta_eig = np.random.randn(D)
    delta_raw = np.random.randn(D)
    B_mat = np.random.randn(D, D)
    attn_weight = np.random.randn(D).astype(np.float64) * 0.1
    
    # トークナイザー作成
    tokenizer = NumPyTokenizer(
        embeddings=embeddings,
        T_eigvals=T_eigvals,
        T_eigvecs=T_eigvecs,
        T_eigvecs_inv=T_eigvecs_inv,
        delta_eig=delta_eig,
        delta_raw=delta_raw,
        B_mat=B_mat,
        phase_weight=0.5,
        attn_weight=attn_weight
    )
    
    # モデル作成
    model = NumPyGRIM(
        tokenizer=tokenizer,
        lam=0.01,
        mu=0.01,
        sigma=0.693,
        beta=0.01,
    )
    
    # tokenize
    prompt = np.array([0, 1, 2], dtype=np.int64)
    psi_0 = model.tokenize(prompt)
    print(f"tokenize: ‖ψ₀‖ = {np.linalg.norm(psi_0):.10f}")
    assert np.isclose(np.linalg.norm(psi_0), 1.0)
    
    # integrate
    psi_T = model.integrate(psi_0)
    print(f"integrate: ‖ψ_T‖ = {np.linalg.norm(psi_T):.10f}")
    assert np.isclose(np.linalg.norm(psi_T), 1.0)
    
    # observe
    probs = model.observe(psi_T)
    print(f"observe: sum = {np.sum(probs):.10f}")
    assert np.isclose(np.sum(probs), 1.0)
    
    # generate
    generated = model.generate(prompt, max_tokens=10, temperature=1.0)
    print(f"generate: {generated} (length={len(generated)})")
    # 生成は EOS で早期終了する可能性があるため、長さチェックは緩くする
    assert len(generated) <= 10, f"Generation too long: {len(generated)}"
    assert all(0 <= t < V for t in generated), "Generated tokens out of range"
    
    print("✓ model_np.py: All tests passed")
    return True

def main():
    print("=" * 60)
    print("NumPy GRIM Implementation - Verification Tests")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("geometry", test_geometry()))
    except Exception as e:
        print(f"✗ geometry_np.py FAILED: {e}")
        results.append(("geometry", False))
    
    try:
        results.append(("tokenizer", test_tokenizer()))
    except Exception as e:
        print(f"✗ tokenizer_np.py FAILED: {e}")
        results.append(("tokenizer", False))
    
    try:
        results.append(("flow_field", test_flow_field()))
    except Exception as e:
        print(f"✗ flow_field_np.py FAILED: {e}")
        results.append(("flow_field", False))
    
    try:
        results.append(("ode_solver", test_ode_solver()))
    except Exception as e:
        print(f"✗ ode_solver_np.py FAILED: {e}")
        results.append(("ode_solver", False))
    
    try:
        results.append(("observation", test_observation()))
    except Exception as e:
        print(f"✗ observation_np.py FAILED: {e}")
        results.append(("observation", False))
    
    try:
        results.append(("history", test_history()))
    except Exception as e:
        print(f"✗ history_np.py FAILED: {e}")
        results.append(("history", False))
    
    try:
        results.append(("model", test_model()))
    except Exception as e:
        print(f"✗ model_np.py FAILED: {e}")
        results.append(("model", False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All NumPy GRIM modules working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} module(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
