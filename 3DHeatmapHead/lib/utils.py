import tensorflow as tf
import numpy as np

class ModelInspector:
    @staticmethod
    def trace(model, input_shape=(224, 224, 3), batch_size=1):
        print(
            f"""
            {"="*50}
            Model Inspector
            {"="*50}
            """
        )

        inputs = tf.random.normal((batch_size, *input_shape))
        current_x = inputs

        print(f"Inputs_channels : {current_x.shape}")
        print("-"*50)

        for i, layer in enumerate(model._layers):
            try:
                shape_in = current_x.get_shape().as_list()
                current_x = layer(current_x, training=False)
                shape_out = current_x.get_shape().as_list()

                params = sum([np.prod(p.shape) for p in layer.trainable_variables])
                print(f"[{i:02d}] {layer.name:<20} | {str(shape_in):<18} -> {str(shape_out):<18} | Params: {params:,}")

            except Exception as e:
                print(f"Crash at layer {i} ({layer.name})")
                print(f"    Error: {e}")
                break

            print("-" * 50)
            print(f"Output_channels : {current_x.shape}")
            print("=" * 50 + "\n")

        @staticmethod
        def count_params(model):
            trainable = sum([np.prod(p.shape) for p in model.trainable_variables])
            non_trainable = sum([np.prod(p.shape) for p in model.non_trainable_variables])
            print(f"Total Params: {trainable + non_trainable:,}")
            print(f"Trainable: {trainable:,}")
