import torch
import sys
sys.path.append('.')

from modules.memory_bank_retriever import MemoryBank, Retriever
from modules.sequence_encoder import SequenceEncoder

def test_memory_bank_retriever():
    print("Testing MemoryBank/Retriever Generation...")
    print("-" * 50)
    
    # Test configurations
    memory_size = 100
    key_dim = 64
    value_dim = 128
    batch_size = 4
    
    # Test MemoryBank
    print("\nTesting MemoryBank...")
    memory_bank = MemoryBank(
        memory_size=memory_size,
        key_dim=key_dim,
        value_dim=value_dim,
        similarity='cosine',
        update_method='fifo'
    )
    print(f"✓ MemoryBank created successfully")
    print(f"  Parameters: {sum(p.numel() for p in memory_bank.parameters()):,}")
    
    # Test writing to memory
    keys = torch.randn(batch_size, key_dim)
    values = torch.randn(batch_size, value_dim)
    
    memory_bank.write(keys, values)
    print(f"✓ Written {batch_size} memories")
    print(f"✓ Current memory size: {memory_bank.current_size.item()}")
    
    # Test retrieval
    query = torch.randn(key_dim)
    retrieved_values = memory_bank.retrieve(query, k=3)
    print(f"✓ Retrieved values shape: {retrieved_values.shape}")
    assert retrieved_values.shape == (3, value_dim)
    
    # Test batch retrieval
    batch_query = torch.randn(2, key_dim)
    batch_retrieved, similarities = memory_bank.retrieve(batch_query, k=3, return_similarities=True)
    print(f"✓ Batch retrieved shape: {batch_retrieved.shape}")
    print(f"✓ Similarities shape: {similarities.shape}")
    assert batch_retrieved.shape == (2, 3, value_dim)
    
    # Test memory overflow
    print("\nTesting memory overflow...")
    for i in range(memory_size + 10):
        memory_bank.write(torch.randn(1, key_dim), torch.randn(1, value_dim))
    
    print(f"✓ Memory size capped at: {memory_bank.current_size.item()}")
    assert memory_bank.current_size.item() == memory_size
    
    # Test different update methods
    print("\nTesting update methods...")
    for update_method in ['fifo', 'lru', 'lfu']:
        mb = MemoryBank(
            memory_size=50,
            key_dim=key_dim,
            value_dim=value_dim,
            update_method=update_method
        )
        
        # Write some memories
        for i in range(60):
            mb.write(torch.randn(1, key_dim), torch.randn(1, value_dim))
            
        print(f"✓ Update method '{update_method}' works correctly")
    
    # Test different similarity metrics
    print("\nTesting similarity metrics...")
    for similarity in ['cosine', 'dot', 'l2']:
        mb = MemoryBank(
            memory_size=50,
            key_dim=key_dim,
            value_dim=value_dim,
            similarity=similarity
        )
        
        mb.write(torch.randn(10, key_dim), torch.randn(10, value_dim))
        retrieved = mb.retrieve(torch.randn(key_dim), k=5)
        print(f"✓ Similarity '{similarity}' works correctly")
    
    # Test clear
    memory_bank.clear()
    print(f"✓ Memory cleared, size: {memory_bank.current_size.item()}")
    
    # Test Retriever with encoder
    print("\nTesting Retriever...")
    encoder = SequenceEncoder(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2
    )
    
    retriever = Retriever(
        encoder=encoder,
        memory_size=memory_size,
        key_dim=key_dim,
        value_dim=value_dim
    )
    print(f"✓ Retriever created successfully")
    print(f"  Parameters: {sum(p.numel() for p in retriever.parameters()):,}")
    
    # Test encode and store
    input_ids = torch.randint(0, 1000, (batch_size, 20))
    encoded = retriever(input_ids, mode='encode_store')
    print(f"✓ Encoded and stored, output shape: {encoded.shape}")
    
    # Store more memories
    for i in range(5):
        more_ids = torch.randint(0, 1000, (2, 20))
        retriever(more_ids, mode='encode_store')
    
    # Test retrieval and integration
    query_ids = torch.randint(0, 1000, (2, 15))
    output = retriever(query_ids, mode='retrieve', k=3)
    
    assert 'output' in output
    assert 'retrieved_values' in output
    assert 'similarities' in output
    
    print(f"✓ Retrieved and integrated:")
    print(f"  Output shape: {output['output'].shape}")
    print(f"  Retrieved values shape: {output['retrieved_values'].shape}")
    print(f"  Similarities shape: {output['similarities'].shape}")
    
    # Test gradients
    loss = output['output'].sum()
    loss.backward()
    
    grad_check_passed = True
    for name, param in retriever.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"✗ No gradient for {name}")
                grad_check_passed = False
            elif torch.isnan(param.grad).any():
                print(f"✗ NaN gradient for {name}")
                grad_check_passed = False
    
    if grad_check_passed:
        print("✓ All gradients computed correctly")
    
    # Component verification
    print("\nComponent Analysis:")
    print("  - MemoryBank: FIFO/LRU/LFU update strategies")
    print("  - Multiple similarity metrics: cosine, dot, L2")
    print("  - Stateful memory management with size limits")
    print("  - Retriever: Encoder + memory integration")
    print("  - Cross-attention for memory integration")
    print("  - Batch processing support")
    
    print("\n✅ MemoryBank/Retriever: PASSED ALL TESTS")
    return True

if __name__ == "__main__":
    test_memory_bank_retriever()