import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Load your pretrained model
loaded_model = tf.keras.models.load_model("D:/Josika/Metaverse/sum.h5")

# Define pruning parameters
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,  # Initial sparsity (0.0 means no pruning)
        final_sparsity=0.5,    # Final desired sparsity level
        begin_step=0,          # Step at which pruning begins
        end_step=10000         # Step at which pruning ends (adjust as needed)
    )
}

# Apply weight pruning to the loaded model
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(loaded_model, **pruning_params)

# Compile the pruned model
pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=  ['accuracy', 'precision', 'recall', 'f1_score'])

# Save the pruned model
pruned_model.save('llm_pruned_model.h5')