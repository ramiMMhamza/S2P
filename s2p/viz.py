"""Optional TensorBoard visualisations (``S2P.log_images: True``)."""

import torch
import torchvision
import torchvision.transforms as T
from openunreid.data.utils.data_utils import read_image
from torchvision.transforms import InterpolationMode

__all__ = ["log_support_pairs"]


def log_support_pairs(writer, task_id, target_set, source_set, selection, max_pairs=16):
    """Show the (target, nearest source) pairs with the lowest and highest similarity."""
    if writer is None or len(selection.pairs) == 0:
        return
    to_tensor = T.Compose(
        [T.Resize((256, 128), interpolation=InterpolationMode.BICUBIC), T.ToTensor()]
    )
    pairs = sorted(selection.pairs, key=lambda pair: pair[2])
    for tag, subset in (("least_similar", pairs[:max_pairs]), ("most_similar", pairs[-max_pairs:])):
        tiles = []
        for target_index, source_index, _ in subset:
            target_image = to_tensor(read_image(target_set.data[target_index][0]))
            source_image = to_tensor(read_image(source_set.data[source_index][0]))
            tiles.append(torch.cat((target_image, source_image), dim=2))
        grid = torchvision.utils.make_grid(tiles, nrow=8)
        writer.add_image(f"support_pairs/{tag}", grid, task_id)
    similarities = torch.tensor([pair[2] for pair in selection.pairs])
    writer.add_scalar("support_pairs/mean_similarity", similarities.mean().item(), task_id)
