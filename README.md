# ResPA
This repository contains the  code for the ResPA.

If you are interested in the paper and the code, please contact us at zeze@hbu.edu.cn 





## Qucik Start
### Prepare the dataset and models.
1. You can download the ImageNet-compatible dataset and put the data in **'./dataset/'**.

2. The normally trained models (i.e., Inc-v3, Res-50, Den-121) are from "pretrainedmodels", if you use it for the first time, it will download the weight of the model automatically, just wait for it to finish. 

3. The adversarially trained models (i.e, ens3_adv_inc_v3, ens4_adv_inc_v3, ens_adv_inc_res_v2) are from [SSA](https://github.com/yuyang-long/SSA) or [tf_to_torch_model](https://github.com/ylhz/tf_to_pytorch_model). For more detailed information on how to use them, visit these two repositories.

### Runing attack
1. You can run our proposed attack as follows. 
```
python Incv3_PGN_Attack.py
```
2. The generated adversarial examples would be stored in the directory **./incv3_xx_xx_outputs**. Then run the file **verify.py** to evaluate the attack success rate of each model used in the paper:
```
python verify.py
```
3. You can run the file **'surface_map.py'** to visualize the loss surface maps for the adversarial examples, the maps will be stored in the directory **'./loss_surfaces/'**.
```
python surface_map.py
```
## Citation
If our paper or this code is useful for your research, please cite our paper.
```
The details of our paper will be updated shortly.
```
