import torch

def clip_contrastive_loss(image_embeds, text_embeds, logit_scale):
    """
    image_embeds : (N, d), already L2-normalized (CLIP.forward does this)
    text_embeds : (N, d), already L2-normalized (CLIP.forward does this)
    image_to_text, text_to_image : (N, N)
    """
    image_to_text = logit_scale * (image_embeds @ text_embeds.T)
    text_to_image = image_to_text.T

    log_probs_i = torch.log_softmax(image_to_text, dim=-1)
    log_probs_t = torch.log_softmax(text_to_image, dim=-1)

    loss_i = -log_probs_i.diagonal().mean()
    loss_t = -log_probs_t.diagonal().mean()

    return (loss_i + loss_t) / 2
