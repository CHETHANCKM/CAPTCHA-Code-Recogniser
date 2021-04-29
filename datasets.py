import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_files("fournierp/captcha-version-2-images",
									path="Datasets/",
									unzip=True)