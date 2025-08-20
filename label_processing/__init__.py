import warnings

# Suppress external library warnings
warnings.filterwarnings('ignore', message='The parameter \'pretrained\' is deprecated.*')
warnings.filterwarnings('ignore', message='Arguments other than a weight enum.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', module='urllib3')
warnings.filterwarnings('ignore', module='huggingface_hub')
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL.*')
