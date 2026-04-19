import pandas as pd
from trainer import run_qcl, QCLRunConfig

def main():
    print("Starting QCL Ablation Study...")
    print("Evaluating architecture impact using 'ideal' simulator and 'scratch' initialization.")
    print("Using 1 epoch on full dataset to check structural topology limit quickly.\n", flush=True)
    
    qubit_list = [2, 4, 8]
    layer_list = [1, 2, 4]
    results = []

    for n_q in qubit_list:
        for n_l in layer_list:
            run_id = f"ablation_sq__q{n_q}_l{n_l}"
            print(f"--- Running Config: n_qubits={n_q}, n_layers={n_l} ---", flush=True)
            cfg = QCLRunConfig(
                run_id=run_id,
                ansatz="strongly_entangling",
                noise_model="ideal",
                source="scratch",
                seed=42,
                n_qubits=n_q,
                n_layers=n_l,
                epochs=1,           # Just 1 epoch on full dataset to measure structural limit
                pretrain_epochs=1,
                batch_size=256,     # Increased batch size to speed up local testing
                freeze_prior=False
            )
            
            # Execute the QCL loop directly
            try:
                res = run_qcl(cfg)
                # Store results for the table
                results.append({
                    "n_qubits": n_q,
                    "n_layers": n_l,
                    "Acc_A_Init": f"{res.acc_a_initial*100:.1f}%",
                    "Acc_B": f"{res.acc_b_final*100:.1f}%",
                    "Acc_A_Final": f"{res.acc_a_final*100:.1f}%",
                    "Total Time": f"{(res.train_time_a_s + res.train_time_b_s):.1f}s"
                })
            except Exception as e:
                print(f"Error evaluating q={n_q}, l={n_l}: {e}")

    df = pd.DataFrame(results)
    print("==================================", flush=True)
    print("ABLATION STUDY RESULTS (IDEAL)", flush=True)
    print("==================================", flush=True)
    print(df.to_markdown(index=False), flush=True)

if __name__ == "__main__":
    main()
