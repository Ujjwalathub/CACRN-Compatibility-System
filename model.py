import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

def build_model(base_dim=1154, logic_dim=12):
    # ==========================================================================
    # INPUTS
    # ==========================================================================
    # 1. Base Inputs (The 1154 Text/Num Features)
    input_src = layers.Input(shape=(base_dim,), name="src_input")
    input_dst = layers.Input(shape=(base_dim,), name="dst_input")
    
    # 2. Logic Inputs (The 12 Handcrafted Features)
    input_logic = layers.Input(shape=(logic_dim,), name="logic_input")

    # ==========================================================================
    # SHARED ENCODER (The "Siamese" Branch)
    # ==========================================================================
    # Encoder architecture from saved weights:
    # Dense(1154→128) + BN + Dropout → Dense(128→64) + BN + Dropout → Dense(64→32)
    encoder = models.Sequential([
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu')
    ], name="encoder")
    
    encoded_src = encoder(input_src)
    encoded_dst = encoder(input_dst)

    # ==========================================================================
    # INTERACTION LAYER
    # ==========================================================================
    # 1. Dot Product
    dot = layers.Dot(axes=1)([encoded_src, encoded_dst])
    
    # 2. Concatenation
    concat = layers.Concatenate()([encoded_src, encoded_dst])
    
    # 3. Absolute Difference
    diff = layers.Lambda(lambda x: K.abs(x[0] - x[1]))([encoded_src, encoded_dst])
    
    # 4. Multiply
    mult = layers.Multiply()([encoded_src, encoded_dst])
    
    # Reshape dot product to align dimensions
    dot = layers.Reshape((1,))(dot)

    # ==========================================================================
    # MERGE ALL PATHS
    # ==========================================================================
    # Combine Deep Interactions + Logic Features
    # This structure must match the "concatenate_1" from your logs
    merged = layers.Concatenate()([concat, diff, mult, dot, input_logic])

    # ==========================================================================
    # DENSE LAYERS (The Classifier)
    # ==========================================================================
    # These are Layers 2-8 in your saved file
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Dense(16, activation='relu')(x)
    
    # Output
    output = layers.Dense(1, activation='sigmoid', name="score")(x)

    model = models.Model(
        inputs=[input_src, input_dst, input_logic], 
        outputs=output
    )
    
    return model
