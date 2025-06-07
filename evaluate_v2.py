import json
import numpy as np
from tinygrad.tensor import Tensor
from model_v2 import ReimbursementModelV2
import os
from datetime import datetime

def load_and_evaluate_cases(model, cases_file, stats_file):

    isPrivate = cases_file == 'private_cases.json'
    # Load normalization stats
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    mean = np.array(stats['mean'])
    std = np.array(stats['std'])
    
    # Load test cases
    with open(cases_file, 'r') as f:
        cases = json.load(f)
    
    # Prepare batch data
    inputs = []
    expected = []
    has_expected = False
    
    for case in cases:
        if isPrivate:
            inputs.append([
                case['trip_duration_days'],
                case['miles_traveled'],
                case['total_receipts_amount']
            ])
        else:
            inputs.append([
                case['input']['trip_duration_days'],
                case['input']['miles_traveled'],
                case['input']['total_receipts_amount']
            ])
        if 'expected_output' in case:
            has_expected = True
            expected.append(case['expected_output'])
    
    # Convert to numpy arrays
    X = np.array(inputs, dtype=np.float32)
    if has_expected:
        y_true = np.array(expected, dtype=np.float32)
    
    # Normalize inputs
    X_normalized = (X - mean) / std
    
    # Make batch prediction
    Tensor.training = False
    predictions = model(Tensor(X_normalized)).numpy().flatten()
    
    # Calculate metrics and prepare results
    results = {
        'total_cases': len(cases),
        'predictions': [float(p) for p in predictions],
    }
    
    if has_expected:
        errors = np.abs(predictions - y_true)
        exact_matches = np.sum(errors < 0.01)
        close_matches = np.sum(errors < 1.0)
        mae = np.mean(errors)
        max_error = np.max(errors)
        
        # Create detailed case analysis
        all_cases = []
        for idx in range(len(cases)):
            case_info = {
                'case_num': idx + 1,
                'input': {
                    'trip_duration': float(inputs[idx][0]),
                    'miles': float(inputs[idx][1]),
                    'receipts': float(inputs[idx][2])
                },
                'expected': float(y_true[idx]),
                'predicted': float(predictions[idx]),
                'error': float(errors[idx]),
                'is_exact_match': bool(errors[idx] < 0.01),
                'is_close_match': bool(errors[idx] < 1.0)
            }
            all_cases.append(case_info)
        
        # Sort cases by error (descending)
        all_cases.sort(key=lambda x: x['error'], reverse=True)
        
        # Calculate error distribution
        error_distribution = {
            'exact_match': len([c for c in all_cases if c['is_exact_match']]),
            'close_match': len([c for c in all_cases if c['is_close_match'] and not c['is_exact_match']]),
            'error_ranges': {
                '1_to_5': len([c for c in all_cases if 1.0 <= c['error'] < 5.0]),
                '5_to_10': len([c for c in all_cases if 5.0 <= c['error'] < 10.0]),
                'over_10': len([c for c in all_cases if c['error'] >= 10.0])
            }
        }
        
        results.update({
            'exact_matches': int(exact_matches),
            'exact_match_pct': (exact_matches / len(cases)) * 100,
            'close_matches': int(close_matches),
            'close_match_pct': (close_matches / len(cases)) * 100,
            'mae': float(mae),
            'max_error': float(max_error),
            'error_distribution': error_distribution,
            'all_cases': all_cases
        })
    else:
        # For cases without expected values (private cases)
        all_cases = []
        for idx in range(len(cases)):
            case_info = {
                'case_num': idx + 1,
                'input': {
                    'trip_duration': float(inputs[idx][0]),
                    'miles': float(inputs[idx][1]),
                    'receipts': float(inputs[idx][2])
                },
                'predicted': float(predictions[idx])
            }
            all_cases.append(case_info)
        results['all_cases'] = all_cases
    
    return results

def save_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    # Force CPU execution
    os.environ["CPU"] = "1"
    
    print("🧾 Evaluating model...")
    
    # Load model
    model = ReimbursementModelV2()
    model.load('best_model_v2.npy')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_file': 'best_model_v2.npy'
    }
    
    # Evaluate public cases
    print("Processing public cases...")
    public_results = load_and_evaluate_cases(
        model, 
        'public_cases.json',
        'normalization_stats_v2.json'
    )
    results['public_cases'] = public_results
    
    # Evaluate private cases
    print("Processing private cases...")
    private_results = load_and_evaluate_cases(
        model, 
        'private_cases.json',
        'normalization_stats_v2.json'
    )
    results['private_cases'] = private_results
    
    # Save all results to JSON
    output_file = f'evaluation_results_{timestamp}.json'
    save_results(results, output_file)
    
    # Print brief summary
    print("\n✅ Evaluation complete!")
    print(f"📊 Results Summary:")
    print(f"  Public cases processed: {public_results['total_cases']}")
    if 'exact_matches' in public_results:
        print(f"  - Exact matches (±$0.01): {public_results['exact_matches']} ({public_results['exact_match_pct']:.1f}%)")
        print(f"  - Close matches (±$1.00): {public_results['close_matches']} ({public_results['close_match_pct']:.1f}%)")
        print(f"  - Average error: ${public_results['mae']:.2f}")
    print(f"  Private cases processed: {private_results['total_cases']}")
    print(f"💾 Full results saved to: {output_file}")

if __name__ == "__main__":
    main() 
