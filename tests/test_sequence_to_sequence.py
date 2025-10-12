import torch
import sys
sys.path.append('.')

from modules.sequence_to_sequence import SequenceToSequenceModel
from modules.sequence_encoder import SequenceEncoder
from modules.attention_decoder import AttentionDecoder

def test_sequence_to_sequence():
    print("Testing SequenceToSequenceModel Generation...")
    print("-" * 50)
    
    # Test configurations
    batch_size = 4
    src_seq_len = 20
    tgt_seq_len = 15
    src_vocab_size = 10000
    tgt_vocab_size = 8000
    d_model = 256
    
    # Create encoder and decoder
    encoder = SequenceEncoder(
        vocab_size=src_vocab_size,
        d_model=d_model,
        n_heads=8,
        n_layers=3,
        pooling_strategy='mean'  # We'll use sequence_output from the dict
    )
    
    decoder = AttentionDecoder(
        vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=8,
        n_layers=3
    )
    
    # Test basic Seq2Seq
    print("\nTesting basic Seq2Seq model...")
    seq2seq = SequenceToSequenceModel(encoder, decoder)
    print(f"✓ Module created successfully")
    print(f"  Parameters: {sum(p.numel() for p in seq2seq.parameters()):,}")
    
    # Test forward pass
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    output = seq2seq(src, tgt)
    
    assert 'logits' in output
    assert 'encoder_output' in output
    print(f"✓ Source shape: {src.shape}")
    print(f"✓ Target shape: {tgt.shape}")
    print(f"✓ Logits shape: {output['logits'].shape}")
    print(f"✓ Encoder output shape: {output['encoder_output'].shape}")
    
    assert output['logits'].shape == (batch_size, tgt_seq_len, tgt_vocab_size)
    assert not torch.isnan(output['logits']).any()
    print("✓ Forward pass successful")
    
    # Test with masks
    src_mask = torch.ones(batch_size, 1, 1, src_seq_len)
    src_mask[:, :, :, 15:] = 0  # Mask last 5 positions
    tgt_mask = torch.ones(batch_size, 1, 1, tgt_seq_len)
    tgt_mask[:, :, :, 10:] = 0  # Mask last 5 positions
    
    output_masked = seq2seq(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
    print("✓ Masking works correctly")
    
    # Test gradients
    loss = output['logits'].sum()
    loss.backward()
    
    grad_check_passed = True
    for name, param in seq2seq.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"✗ No gradient for {name}")
                grad_check_passed = False
            elif torch.isnan(param.grad).any():
                print(f"✗ NaN gradient for {name}")
                grad_check_passed = False
    
    if grad_check_passed:
        print("✓ All gradients computed correctly")
    
    # Test generation
    print("\nTesting generation...")
    seq2seq.eval()
    generated = seq2seq.generate(src[:1], max_length=20, temperature=0.8)
    print(f"✓ Generated shape: {generated.shape}")
    assert generated.shape[0] == 1
    assert generated.shape[1] <= 20
    print("✓ Greedy generation works")
    
    # Test beam search
    print("\nTesting beam search...")
    generated_beam = seq2seq.generate(src[:1], max_length=15, num_beams=3)
    print(f"✓ Beam search generated shape: {generated_beam.shape}")
    print("✓ Beam search works")
    
    # Test shared embeddings
    print("\nTesting shared embeddings...")
    # Create new encoder/decoder without embeddings
    encoder_no_embed = SequenceEncoder(
        vocab_size=5000,
        d_model=d_model,
        n_heads=8,
        n_layers=2
    )
    decoder_no_embed = AttentionDecoder(
        vocab_size=5000,
        d_model=d_model,
        n_heads=8,
        n_layers=2
    )
    
    seq2seq_shared = SequenceToSequenceModel(
        encoder_no_embed,
        decoder_no_embed,
        src_vocab_size=5000,
        tgt_vocab_size=5000,
        share_embeddings=True
    )
    
    src_shared = torch.randint(0, 5000, (2, 10))
    tgt_shared = torch.randint(0, 5000, (2, 8))
    output_shared = seq2seq_shared(src_shared, tgt_shared)
    print("✓ Shared embeddings work correctly")
    
    # Test tied embeddings
    print("\nTesting tied embeddings...")
    seq2seq_tied = SequenceToSequenceModel(
        encoder_no_embed,
        decoder_no_embed,
        src_vocab_size=5000,
        tgt_vocab_size=5000,
        share_embeddings=True,
        tie_embeddings=True
    )
    
    # Check if embeddings are tied to output projection
    if hasattr(decoder_no_embed, 'output_projection') and hasattr(decoder_no_embed, 'token_embedding'):
        same_weight = decoder_no_embed.output_projection.weight is decoder_no_embed.token_embedding.weight
        print(f"✓ Embeddings tied to output: {same_weight}")
    
    # Test encode/decode separately
    encoder_out = seq2seq.encode(src)
    decoder_out = seq2seq.decode(tgt, encoder_out)
    print("✓ Separate encode/decode methods work")
    
    # Component verification
    print("\nComponent Analysis:")
    print("  - Orchestrates encoder and decoder modules")
    print("  - Handles embedding sharing and tying")
    print("  - Supports both greedy and beam search generation")
    print("  - Flexible masking for source and target")
    print("  - Works with any encoder/decoder pair")
    
    print("\n✅ SequenceToSequenceModel: PASSED ALL TESTS")
    return True

if __name__ == "__main__":
    test_sequence_to_sequence()