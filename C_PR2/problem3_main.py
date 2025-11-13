import math

# Physical parameters
rho = 300  # kg/m³
r1 = r2 = r3 = 0.02  # m
l1 = l2 = l3 = 0.2  # m
g = 9.81  # m/s²

# Calculate mass for each link (cylindrical links)
def calculate_masses():
    m1 = rho * math.pi * r1**2 * l1
    m2 = rho * math.pi * r2**2 * l2
    m3 = rho * math.pi * r3**2 * l3
    
    return m1, m2, m3

m1, m2, m3 = calculate_masses()

def compute_inertia_matrix(q):
    q1, q2, q3 = q
    
    c2 = math.cos(q2)
    s2 = math.sin(q2)
    c3 = math.cos(q3)
    s3 = math.sin(q3)
    c23 = math.cos(q2 + q3)
    
    D11 = (m1/3 + m2 + m3) * l1**2 + \
          (m2/3 + m3) * l2**2 + m3 * l3**2/3 + \
          2*m2*l1*l2*c2 + 2*m3*l1*l2*c2 + \
          2*m3*l1*l3*c23 + 2*m3*l2*l3*c3
    
    D12 = (m2/3 + m3) * l2**2 + m3 * l3**2/3 + \
          m2*l1*l2*c2 + m3*l1*l2*c2 + \
          m3*l1*l3*c23 + 2*m3*l2*l3*c3
    
    D13 = m3 * l3**2/3 + m3*l1*l3*c23 + m3*l2*l3*c3
    
    D22 = (m2/3 + m3) * l2**2 + m3*l3**2/3 + 2*m3*l2*l3*c3
    
    D23 = m3 * l3**2/3 + m3*l2*l3*c3
    
    D33 = m3 * l3**2 / 3
    
    D = [
        [D11, D12, D13],
        [D12, D22, D23],
        [D13, D23, D33]
    ]
    
    return D

def compute_coriolis_centrifugal(q, dq):
    q1, q2, q3 = q
    dq1, dq2, dq3 = dq
    
    s2 = math.sin(q2)
    s3 = math.sin(q3)
    s23 = math.sin(q2 + q3)
    
    c1 = -m2*l1*l2*s2*(2*dq1*dq2 + dq2**2) - \
         m3*l1*l2*s2*(2*dq1*dq2 + dq2**2) - \
         m3*l1*l3*s23*(2*dq1*(dq2 + dq3) + (dq2 + dq3)**2) - \
         2*m3*l2*l3*s3*(dq1*dq3 + dq2*dq3 + dq3**2)
    
    c2 = m2*l1*l2*s2*dq1**2 + \
         m3*l1*l2*s2*dq1**2 + \
         m3*l1*l3*s23*dq1**2 - \
         2*m3*l2*l3*s3*dq3*(dq2 + dq3) - \
         m3*l2*l3*s3*dq3**2
    
    c3 = m3*l1*l3*s23*dq1**2 + \
         m3*l2*l3*s3*(dq1**2 + 2*dq1*dq2 + dq2**2)
    
    C_dq = [c1, c2, c3]
    
    return C_dq

def compute_gravity_terms(q):
    q1, q2, q3 = q
    
    g1 = -(m1/2 + m2 + m3) * g * l1 * math.sin(q1) - \
         (m2/2 + m3) * g * l2 * math.sin(q1 + q2) - \
         m3/2 * g * l3 * math.sin(q1 + q2 + q3)
    
    g2 = -(m2/2 + m3) * g * l2 * math.sin(q1 + q2) - \
         m3/2 * g * l3 * math.sin(q1 + q2 + q3)
    
    g3 = -m3/2 * g * l3 * math.sin(q1 + q2 + q3)
    
    g_vec = [g1, g2, g3]
    
    return g_vec

def print_matrix(M, name="Matrix"):
    print(f"\n{name}:")
    for row in M:
        if isinstance(row, list):
            print("  [" + ", ".join([f"{val:12.6e}" for val in row]) + "]")
        else:
            print(f"  {row:12.6e}")

def matrix_determinant_3x3(M):
    return (M[0][0] * (M[1][1]*M[2][2] - M[1][2]*M[2][1]) -
            M[0][1] * (M[1][0]*M[2][2] - M[1][2]*M[2][0]) +
            M[0][2] * (M[1][0]*M[2][1] - M[1][1]*M[2][0]))

def test_configuration(q, dq, config_name=""):
    print(f"Configuration {config_name}:")
    print(f"  q  = [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}] rad")
    print(f"  dq = [{dq[0]:.4f}, {dq[1]:.4f}, {dq[2]:.4f}] rad/s")
    
    D = compute_inertia_matrix(q)
    C_dq = compute_coriolis_centrifugal(q, dq)
    g_vec = compute_gravity_terms(q)
    
    print_matrix(D, "Inertia Matrix D(q)")
    det_D = matrix_determinant_3x3(D)
    print(f"  Determinant: {det_D:.6e}")
    
    print_matrix(C_dq, "Coriolis/Centrifugal C(q,dq)*dq")
    print_matrix(g_vec, "Gravity g(q)")
    
    return D, C_dq, g_vec

# Test with three different configurations
if __name__ == "__main__":
    print("Problem 3")
    print(f"  Masses: m1={m1:.6f} kg, m2={m2:.6f} kg, m3={m3:.6f} kg")
    print(f"  Lengths: l1={l1} m, l2={l2} m, l3={l3} m")
    print(f"  Radii: r1={r1} m, r2={r2} m, r3={r3} m")
    print(f"  Density: ρ={rho} kg/m³")
    print(f"  Gravity: g={g} m/s²")
    
    q1 = [0.0, 0.0, 0.0]
    dq1 = [0.0, 0.0, 0.0]
    D1, C1, g1 = test_configuration(q1, dq1, "1 (all zeros)")
    
    q2 = [math.pi/4, math.pi/6, math.pi/3]
    dq2 = [0.5, 0.3, 0.2]
    D2, C2, g2 = test_configuration(q2, dq2, "2 (non-zero)")
    
    q3 = [math.pi/2, -math.pi/4, math.pi/6]
    dq3 = [1.0, -0.5, 0.8]
    D3, C3, g3 = test_configuration(q3, dq3, "3 (mixed)")
    
    print("Analysis complete!")

