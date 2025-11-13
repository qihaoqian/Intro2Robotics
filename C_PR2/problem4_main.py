import math
from problem3_main import compute_gravity_terms

def print_result(config_num, q, tau):
    print(f"\nConfiguration {config_num}:")
    print(f"  θ = [{q[0]/math.pi:.4f}π, {q[1]/math.pi:.4f}π, {q[2]/math.pi:.4f}π] rad")
    print(f"    = [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}] rad")
    print(f"\n  Required Torque τ = g(q):")
    print(f"    τ₁ = {tau[0]:10.6f} N·m")
    print(f"    τ₂ = {tau[1]:10.6f} N·m")
    print(f"    τ₃ = {tau[2]:10.6f} N·m")
    print(f"\n  Vector form: τ = [{tau[0]:.6f}, {tau[1]:.6f}, {tau[2]:.6f}]")

if __name__ == "__main__":
    print("Problem 4")
    
    q1 = [0.0, 0.0, math.pi/2]
    tau1 = compute_gravity_terms(q1)
    print_result(1, q1, tau1)
    
    q2 = [math.pi/2, math.pi/2, math.pi/4]
    tau2 = compute_gravity_terms(q2)
    print_result(2, q2, tau2)
    
    q3 = [math.pi/3, math.pi/2, 0.0]
    tau3 = compute_gravity_terms(q3)
    print_result(3, q3, tau3)
    
    print("Results:")
    print(f"Config 1: θ = [0, 0, π/2]")
    print(f"  τ = [{tau1[0]:.6f}, {tau1[1]:.6f}, {tau1[2]:.6f}] N·m")
    print(f"Config 2: θ = [π/2, π/2, π/4]")
    print(f"  τ = [{tau2[0]:.6f}, {tau2[1]:.6f}, {tau2[2]:.6f}] N·m")
    print(f"Config 3: θ = [π/3, π/2, 0]")
    print(f"  τ = [{tau3[0]:.6f}, {tau3[1]:.6f}, {tau3[2]:.6f}] N·m")
