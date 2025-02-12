from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from source.model import LightningSimpleMLP
from source.data import QNoiseData
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
import numpy as np
from source.utils import set_seed

set_seed(42)

n_samples = 10000
batch_size = 256
val_test_split = (0., 0.)
num_qubits = 1
num_layers = 1

# DataModule (no need to call setup)
datamodule = QNoiseData(n_samples, batch_size, val_test_split, num_qubits, num_layers)
datamodule.setup()

# x = datamodule.x
# y = datamodule.y

# # Plot data
# x = np.array([i.numpy() for i in x])
# y = np.array(y)
# plt.scatter(x[:, 0], x[:, 1], c=y[:,0])
# plt.xlabel("z1")
# plt.ylabel("z2")
# plt.title("Data")
# plt.show()

hidden_dims = [1, 2, 8, 16, 32, 64]
lr = 0.001
results = {}

for hidden_dim in hidden_dims:
    model = LightningSimpleMLP(input_dim=2,
                            n_layers=2,
                            hidden_dim=hidden_dim,
                            output_dim=num_qubits,
                            dropout=0.,
                            lr=lr)

    callbacks = [ModelCheckpoint(monitor='train_loss'),
                EarlyStopping(monitor='train_loss', patience=5)]

    trainer = Trainer(max_epochs=100, callbacks=callbacks)

    # Train model
    trainer.fit(model, datamodule.train_dataloader())#, datamodule.train_dataloader())

    # Load best checkpoint safely
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path is not None:
        model = LightningSimpleMLP.load_from_checkpoint(best_model_path)
        res = trainer.test(model, datamodule.train_dataloader())
        results[hidden_dim] = res[0]
        results[hidden_dim]['n_params'] = sum(p.numel() for p in model.parameters())
    else:
        print("No best model found, skipping test.")

# Print results
for k, v in results.items():
    print(f"Hidden dim: {k}, N Params: {v['n_params']}, Test Loss: {v['test_loss']}")