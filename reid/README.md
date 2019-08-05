# ReID code for both Plain ReID track and Wild ReID track

## Environment

We are using pytorch 1.1.0 and apex for training our model.

## Start training

- Step 1: `cd evaluate/eval_cylib && make` to compile the Cython code for mAP calculation.
- Step 2: reorganize the data

Note that we did not use all the training images to train our model, we select some image to enable local validation

The id of the image used as local query and gallery set is gaven in the `local_query.txt` and `local_gallery.txt`

Rename all the image to `(qid)_c(cid)_(iid).jpg` format, where the `qid` represent the image's coresponding tiger id,
`cid` is set to be 99 for local train and query set, and 98 for local gallery set, `iid` is the origin image id 

Then put training image into a folder named `local_train`, query to `local_query`, gallery to `local_gallery`, 
and point the path in the config file to them.

- Step 3: run `CUDA_VISIBLE_DEVICES=<gpu-id> python main.py -c configs/cvwc1.yml` to start training

The pretrained model can be downloaded from [here]()

## Inference

After the training is finished, use the `cvwc_submit.ipynb` for plain ReID inference, `vis.ipynb` for detection_test data
preprocess, and `cvwc_wild.ipynb` for Wild ReID inference.

