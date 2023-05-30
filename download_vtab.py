
import tensorflow_datasets as tfds
data_dir = "./datasets/vtab_data"  # TODO: setup the data_dir to put the the data to, the DATA.DATAPATH value in config

# # caltech101 报错 tensorflow.python.framework.errors_impl.NotFoundError: Could not find directory datasets/vtab_data/downloads/ucexport_download_id_137RyRjvTBkBiIfeYBNZ_Ewspr27OLzOXkcog-FWUPYtV3WCJLAolEF_NYx7j1kMPmSY
# dataset_builder = tfds.builder("caltech101:3.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()

# # cifar100 done
# dataset_builder = tfds.builder("cifar100:3.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()
#
# # clevr done
# dataset_builder = tfds.builder("clevr:3.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()
#
# # dmlab done
# dataset_builder = tfds.builder("dmlab:2.0.1", data_dir=data_dir)
# dataset_builder.download_and_prepare()
#
# # dsprites done
# dataset_builder = tfds.builder("dsprites:2.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()
#
# # dtd done
# dataset_builder = tfds.builder("dtd:3.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()
#
# eurosat done
# subset="rgb"
# dataset_name = "eurosat/{}:2.*.*".format(subset)
# dataset_builder = tfds.builder(dataset_name, data_dir=data_dir)
# dataset_builder.download_and_prepare()
#
# # oxford_flowers102 done
# dataset_builder = tfds.builder("oxford_flowers102:2.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()
#
# # oxford_iiit_pet done
# dataset_builder = tfds.builder("oxford_iiit_pet:3.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()
#
# # patch_camelyon tensorflow_datasets.core.download.extractor.ExtractError: Error while extracting datasets/vtab_data/downloads/zeno.org_reco_2546_file_came_leve_2_spli_tr1hnnQUaKerNcfkp15oIbfn5slBFwXUVwjyoO_IlgZWw.gz to datasets/vtab_data/downloads/extracted/GZIP.zeno.org_reco_2546_file_came_leve_2_spli_tr1hnnQUaKerNcfkp15oIbfn5slBFwXUVwjyoO_IlgZWw.gz (file: ) : Compressed file ended before the end-of-stream marker was reached
dataset_builder = tfds.builder("patch_camelyon:2.*.*", data_dir=data_dir)
dataset_builder.download_and_prepare()
#
# # smallnorb done
# dataset_builder = tfds.builder("smallnorb:2.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()
#
# # svhn done
# dataset_builder = tfds.builder("svhn_cropped:3.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()

# raise ReadError("unexpected end of data")
# sun397 --> need cv2
# cannot load one image, similar to issue here: https://github.com/tensorflow/datasets/issues/2889
# "Image /t/track/outdoor/sun_aophkoiosslinihb.jpg could not be decoded by Tensorflow.""
# sol: modify the file: "/fsx/menglin/conda/envs/prompt_tf/lib/python3.7/site-packages/tensorflow_datasets/image_classification/sun.py" to ignore those images
# dataset_builder = tfds.builder("sun397/tfds:4.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()

# kitti done version is wrong from vtab repo, try 3.2.0 (https://github.com/google-research/task_adaptation/issues/18)
# dataset_builder = tfds.builder("kitti:3.2.0", data_dir=data_dir)
# dataset_builder.download_and_prepare()


# diabetic_retinopathy done
"""
Download this dataset from Kaggle.
https://www.kaggle.com/c/diabetic-retinopathy-detection/data
After downloading, 
- unpack the test.zip file into <data_dir>/manual_dir/.
- unpack the sample.zip to sample/. 
- unpack the sampleSubmissions.csv and trainLabels.csv.

# ==== important! ====
# 1. make sure to check that there are 5 train.zip files instead of 4 (somehow if you chose to download all from kaggle, the train.zip.005 file is missing)
# 2. if unzip train.zip ran into issues, try to use jar xvf train.zip to handle huge zip file
cat test.zip.* > test.zip
cat train.zip.* > train.zip
"""

# config_and_version = "btgraham-300" + ":3.*.*"
# dataset_builder = tfds.builder("diabetic_retinopathy_detection/{}".format(config_and_version), data_dir=data_dir)
# dataset_builder.download_and_prepare()


# resisc45 done
"""
download/extract dataset artifacts manually: 
Dataset can be downloaded from OneDrive: https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs
After downloading the rar file, please extract it to the manual_dir.
"""

# dataset_builder = tfds.builder("resisc45:3.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()