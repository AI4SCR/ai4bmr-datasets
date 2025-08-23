from ai4bmr_datasets import BEAT
dm = BEAT()
dm.create_embeddings(model_name='uni_v1', num_workers=12, batch_size=64)