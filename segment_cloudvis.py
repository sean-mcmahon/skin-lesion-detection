from segmenter import *
from cloudvis import CloudVis, Request, Response
PORT = 9999

arch = "/home/sean/Dropbox/Uni/Code/my_rep/cloudvis_worker/example/deploy_col.prototxt"
weights = "/home/sean/hpc-home/Fully-Conv-Network/Resources/FCN_models/cstrip-fcn32s-color/colorSnapshot/_iter_6000.caffemodel"

net = caffe.Net(arch, weights, caffe.TEST)

arch = resnet34
sz = 128
_, tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.NO)
normaliser = tfms.norm
denorm = tfms.denorm
net = build_unet(arch)
weights = '/home/sean/hpc-home/skin_cancer/models/128unet_dermofit_isic17_1.h5'


def run_model(m, x_fn, norm=None):
    im = open_image(x_fn) if isinstance(x_fn, str) else x_fn
    if norm: im = norm(im)
    m.eval()
    if hasattr(m, 'reset'):
        m.reset()
    p = get_prediction(to_np(m(*VV(im))))
    return p, im

def prepImage(img):
    # perform image preprocessing for CNN with caffe

    # no mean subtraction for now
    mean_bgr = np.array((0, 0, 0), dtype=np.float32)
    n_img = np.array(img, dtype=np.float32)
    # make bgr, may not be necessary
    n_img = n_img[:, :, ::-1]
    # mean subtract (nothing subtracted for now)
    n_img -= mean_bgr
    # reshape from w*h*3 to 3*w*h (or 3*h*w??)
    n_img = n_img.transpose((2, 0, 1))

    return n_img


def callback(request, response, data):
    img = request.getImage('input_image')
    overlayFlag = request.getValue('render')

    in_img = prepImage(img)
    # reshap input for any sized image (data blob is N x C x H x W)
    net.blobs['data'].reshape(1, *in_img.shape)
    net.blobs['data'].data[...] = in_img
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
    # give array img values, do i need to convert to BGR?
    out_img = out.astype(np.uint8) * 255
    # cv2.imshow('Image', out_img)
    # cv2.waitKey(0)

    response.addImage('output_image', out_img)
    if overlayFlag:
        # load img as PIL
        img = np.array(img, dtype=np.uint8)
        colorIm = Image.fromarray(img)
        # Network prediction as PIL image
        im = Image.fromarray(out_img, mode='P')
        overlay = Image.blend(colorIm.convert(
            "RGBA"), im.convert("RGBA"), 0.7)
        cv_overlay = np.array(overlay)
        # cv2.imshow('Image2', cv_overlay)
        # cv2.waitKey(0)
        # convert from rgb to bgr - needed?
        # cv_overlay = cv_overlay[:, :, ::-1].copy()
        response.addImage('ouput_overlay_image', cv_overlay)


if __name__ == '__main__':
    # image is in base16 encoding, remove up and including the comma, PNGs start with IV
    dire = ''
    req = Request('{"render": "False", "input_image":  ""  }')

    res = Response()
    #cloudvis = CloudVis(PORT)
    # cloudvis.run(callback)
    callback(req, res, {})
