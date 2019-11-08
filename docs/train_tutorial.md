### Here is the complete command to run ** train.py **:

### 以下是运行 ** train.py ** 的完整命令： 

python train.py --model_mode=slim \
--dataset=./dataset \
--style_image=./style_ims/udnie.jpg \
--VGG_path=./pretrain_model/vgg16.pth \
--image_height=256 \
--image_width=256 \
--learning_rate=1e-3 \
--batch_size=16 \
--epoch=10 \
--style_weight=1e5 \
--content_weight=1e0 \
--tv_weight=1e-7 \
--model_folder=./models/style-transfer

#### Explaination

#### 参数解释

- model_mode: Select the pattern of model, normal or slim. The architecture of paper is normal, slim is the one which is optimized.

  model_mode: 选择模型的模式，normal或者slim。 normal是原论文中的结构， slim是我们优化后的结构 

- dataset: Path to a dataset.

  dataset: 数据集路径。

- style_image: Path to a style image to train.

  style_image: 风格图路径。

- VGG_path: VGG save path.

  VGG_path: VGG 的保存路径。

- image_height: Image's height, which will be fed into model.

  image_height: 图片的高。

- image_width: Image's width, which will be fed into model.

  image_width: 图片的宽。

- learning_rate: A hyperparameter which determines to what extent newly acquired information overrides old information.

  learning_rate: 学习率。

- batch_size: The number of training examples in one forward/backward pass.

  batch_size: 批大小。

- epoch: One cycle through the full training dataset。

  epoch: 总迭代次数。

- style_weight: Hyperparameter, control style learning.

  style_weight: 控制风格学习的超参数。

- content_weight: Hyperparameter, bring into some correspondence with content image.

  content_weight: 控制原图风格学习的超参数。

- tv_weight: Making result image smooth.

  tv_weight: 让生成图片更平滑。

- model_folder: Path to save model.

  model_foler: 模型保存路径。
