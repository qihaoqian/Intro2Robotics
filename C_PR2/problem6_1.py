import math

from problem3_main import l1, l2, l3, g, compute_gravity_terms

def payload_gravity_torque(q, m_payload=3.0):
    q1, q2, q3 = q
    q12 = q1 + q2
    q123 = q12 + q3
    x1 = l1*math.cos(q1) + l2*math.cos(q12) + l3*math.cos(q123)
    x2 =                      l2*math.cos(q12) + l3*math.cos(q123)
    x3 =                                          l3*math.cos(q123)
    tau1 = -m_payload * g * x1
    tau2 = -m_payload * g * x2
    tau3 = -m_payload * g * x3
    return [tau1, tau2, tau3]

def add(v, w):
    return [vi + wi for vi, wi in zip(v, w)]

def print_block(title, q, tau_links, tau_payload, tau_total):
    print(f"\n{title}")
    print(f"  q = [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}] rad"
          f" = [{q[0]/math.pi:.4f}π, {q[1]/math.pi:.4f}π, {q[2]/math.pi:.4f}π]")
    print("  Link gravity only (from Problem 3):")
    print(f"    τ_links = [{tau_links[0]:.6f}, {tau_links[1]:.6f}, {tau_links[2]:.6f}] N·m")
    print("  Added payload gravity (m_p = 3.0 kg at tip):")
    print(f"    τ_payload = [{tau_payload[0]:.6f}, {tau_payload[1]:.6f}, {tau_payload[2]:.6f}] N·m")
    print("  Total required gravity-compensation torque:")
    print(f"    τ_total = [{tau_total[0]:.6f}, {tau_total[1]:.6f}, {tau_total[2]:.6f}] N·m")

if __name__ == "__main__":
    print("Problem 4 (revised): gravity compensation with a 3 kg tip payload\n")
    q_list = [
        ([0.0, 0.0, math.pi/2],          "Configuration 1"),
        ([math.pi/2, math.pi/2, math.pi/4], "Configuration 2"),
        ([math.pi/3, math.pi/2, 0.0],    "Configuration 3"),
    ]
    for q, name in q_list:
        tau_links = compute_gravity_terms(q)
        tau_payload = payload_gravity_torque(q, 3.0)
        tau_total = add(tau_links, tau_payload)
        print_block(name, q, tau_links, tau_payload, tau_total)
