#!/usr/bin/env python3
"""
Audio Quality Analysis Tool using Descript Discriminator
Analyze audio files for quality assessment using state-of-the-art discrimination.
"""

import argparse
import json
import csv
from pathlib import Path
import pandas as pd
from standalone_descript_discriminator import StandaloneDescriptDiscriminator


def main():
    parser = argparse.ArgumentParser(description="Analyze audio quality using Descript discriminator")
    parser.add_argument("input", help="Audio file or directory to analyze")
    parser.add_argument("-o", "--output", help="Output file for results (JSON or CSV)")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Sample rate (default: 44100)")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels (default: 1)")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda (default: cpu)")
    parser.add_argument("--format", choices=["json", "csv", "text"], default="text", 
                       help="Output format (default: text)")
    parser.add_argument("--extensions", nargs="+", 
                       default=[".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"],
                       help="Audio file extensions to process")
    
    args = parser.parse_args()
    
    # Initialize discriminator
    print("🔧 Initializing Descript Discriminator...")
    discriminator = StandaloneDescriptDiscriminator(
        sample_rate=args.sample_rate,
        n_channels=args.channels,
        device=args.device
    )
    
    input_path = Path(args.input)
    
    # Analyze single file or directory
    if input_path.is_file():
        print(f"📄 Analyzing single file: {input_path}")
        results = [discriminator.analyze_file(input_path)]
    elif input_path.is_dir():
        print(f"📁 Analyzing directory: {input_path}")
        results = discriminator.analyze_directory(input_path, args.extensions)
    else:
        print(f"❌ Error: {input_path} is not a valid file or directory")
        return
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', False)]
    
    print(f"\n📊 Analysis Complete:")
    print(f"   ✅ Successful: {len(successful_results)}")
    print(f"   ❌ Failed: {len(failed_results)}")
    
    # Show failures
    if failed_results:
        print(f"\n❌ Failed files:")
        for result in failed_results:
            print(f"   {result['file_name']}: {result['error']}")
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        save_results(successful_results, output_path, args.format)
        print(f"💾 Results saved to: {output_path}")
    
    # Display summary
    if successful_results:
        display_summary(successful_results, args.format)


def save_results(results, output_path, format_type):
    """Save results to file in specified format."""
    
    if format_type == "json":
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    elif format_type == "csv":
        # Flatten results for CSV
        csv_data = []
        for result in results:
            row = {
                'file_name': result['file_name'],
                'file_path': result['file_path'],
                'overall_score': result['overall_score'],
                'audio_length_seconds': result['audio_length_seconds'],
                'num_sub_discriminators': result['num_sub_discriminators'],
            }
            
            # Add individual sub-discriminator scores
            for i, score_info in enumerate(result['scores']):
                row[f'sub_disc_{i}_score'] = score_info['mean_score']
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
    
    elif format_type == "text":
        with open(output_path, 'w') as f:
            f.write("Descript Discriminator Audio Quality Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            for result in results:
                f.write(f"File: {result['file_name']}\n")
                f.write(f"Path: {result['file_path']}\n")
                f.write(f"Overall Score: {result['overall_score']:.4f}\n")
                f.write(f"Duration: {result['audio_length_seconds']:.2f}s\n")
                f.write(f"Sub-discriminator scores:\n")
                
                for score_info in result['scores']:
                    f.write(f"  - Sub-disc {score_info['sub_discriminator']}: {score_info['mean_score']:.4f}\n")
                
                f.write("\n" + "-" * 30 + "\n\n")


def display_summary(results, format_type):
    """Display summary statistics."""
    
    if format_type == "text":
        print(f"\n📈 Summary Statistics:")
        print(f"   Total files analyzed: {len(results)}")
        
        scores = [r['overall_score'] for r in results]
        print(f"   Mean overall score: {sum(scores) / len(scores):.4f}")
        print(f"   Min score: {min(scores):.4f}")
        print(f"   Max score: {max(scores):.4f}")
        
        print(f"\n🏆 Top 5 Quality Files:")
        sorted_results = sorted(results, key=lambda x: x['overall_score'], reverse=True)
        for i, result in enumerate(sorted_results[:5]):
            print(f"   {i+1}. {result['file_name']}: {result['overall_score']:.4f}")
        
        if len(results) > 5:
            print(f"\n⚠️  Bottom 5 Quality Files:")
            for i, result in enumerate(sorted_results[-5:]):
                print(f"   {len(results)-4+i}. {result['file_name']}: {result['overall_score']:.4f}")


def batch_compare_directories():
    """Example function for comparing two directories (e.g., real vs generated audio)."""
    
    print("🔄 Batch Directory Comparison Example")
    print("=" * 40)
    
    # This would compare two directories - useful for evaluating generation quality
    # discriminator = StandaloneDescriptDiscriminator()
    # 
    # real_results = discriminator.analyze_directory("/path/to/real/audio")
    # generated_results = discriminator.analyze_directory("/path/to/generated/audio")
    # 
    # real_scores = [r['overall_score'] for r in real_results if r['success']]
    # generated_scores = [r['overall_score'] for r in generated_results if r['success']]
    # 
    # print(f"Real audio mean score: {sum(real_scores)/len(real_scores):.4f}")
    # print(f"Generated audio mean score: {sum(generated_scores)/len(generated_scores):.4f}")
    # print(f"Quality ratio: {(sum(generated_scores)/len(generated_scores)) / (sum(real_scores)/len(real_scores)):.4f}")


if __name__ == "__main__":
    main()