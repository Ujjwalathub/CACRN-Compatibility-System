import tensorflow as tf
import h5py

# Try to load the model and examine its structure
print("Examining saved model weights structure...")
print("="*80)

try:
    # Method 1: Load full model (may fail but gives us info)
    model = tf.keras.models.load_model('models/best_model.h5', compile=False)
    print("\nModel successfully loaded!")
    print("\nModel Summary:")
    model.summary()
    
    print("\n" + "="*80)
    print("LAYER DETAILS:")
    print("="*80)
    for i, layer in enumerate(model.layers[:15]):
        print(f"\n{i+1}. {layer.name} ({layer.__class__.__name__})")
        print(f"   Total params: {layer.count_params()}")
        if hasattr(layer, 'get_weights') and layer.get_weights():
            for j, w in enumerate(layer.get_weights()):
                print(f"   Weight {j}: shape={w.shape}, dtype={w.dtype}")
                
except Exception as e:
    print(f"\nCouldn't load full model: {e}")
    
# Method 2: Read HDF5 file directly
print("\n" + "="*80)
print("DIRECT HDF5 INSPECTION:")
print("="*80)

with h5py.File('models/best_model.h5', 'r') as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
    
    print("\nModel weights structure:")
    f.visititems(print_structure)
