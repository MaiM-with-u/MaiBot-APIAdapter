import base64
import io

from PIL import Image

from . import _logger as logger
from .payload_content.message import Message, MessageBuilder


def compress_messages(messages: list[Message]) -> list[Message]:
    """
    压缩消息列表中的图片
    :param messages: 消息列表
    :return: 压缩后的消息列表
    """

    def compress_base64_image_by_scale(
        base64_data: str, target_size: int = 0.8 * 1024 * 1024
    ) -> str:
        """压缩base64格式的图片到指定大小
        Args:
            base64_data: base64编码的图片数据
            target_size: 目标文件大小（字节），默认0.8MB
        Returns:
            str: 压缩后的base64图片数据
        """
        try:
            # 将base64转换为字节数据
            image_data = base64.b64decode(base64_data)

            # 如果已经小于目标大小，直接返回原图
            if len(image_data) <= 2 * 1024 * 1024:
                return base64_data

            # 将字节数据转换为图片对象
            img = Image.open(io.BytesIO(image_data))

            # 获取原始尺寸
            original_width, original_height = img.size

            # 计算缩放比例
            scale = min(1.0, (target_size / len(image_data)) ** 0.5)

            # 计算新的尺寸
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            # 创建内存缓冲区
            output_buffer = io.BytesIO()

            # 如果是GIF，处理所有帧
            if getattr(img, "is_animated", False):
                frames = []
                for frame_idx in range(img.n_frames):
                    img.seek(frame_idx)
                    new_frame = img.copy()
                    new_frame = new_frame.resize(
                        (new_width // 2, new_height // 2), Image.Resampling.LANCZOS
                    )  # 动图折上折
                    frames.append(new_frame)

                # 保存到缓冲区
                frames[0].save(
                    output_buffer,
                    format="GIF",
                    save_all=True,
                    append_images=frames[1:],
                    optimize=True,
                    duration=img.info.get("duration", 100),
                    loop=img.info.get("loop", 0),
                )
            else:
                # 处理静态图片
                resized_img = img.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )

                # 保存到缓冲区，保持原始格式
                if img.format == "PNG" and img.mode in ("RGBA", "LA"):
                    resized_img.save(output_buffer, format="PNG", optimize=True)
                else:
                    resized_img.save(
                        output_buffer, format="JPEG", quality=95, optimize=True
                    )

            # 获取压缩后的数据并转换为base64
            compressed_data = output_buffer.getvalue()
            logger.success(
                f"压缩图片: {original_width}x{original_height} -> {new_width}x{new_height}"
            )
            logger.info(
                f"压缩前大小: {len(image_data) / 1024:.1f}KB, 压缩后大小: {len(compressed_data) / 1024:.1f}KB"
            )

            return base64.b64encode(compressed_data).decode("utf-8")

        except Exception as e:
            logger.error(f"压缩图片失败: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return base64_data

    compressed_messages = []
    for message in messages:
        if isinstance(message.content, list):
            # 检查content，如有图片则压缩
            message_builder = MessageBuilder()
            for content_item in message.content:
                if isinstance(content_item, tuple):
                    # 图片，进行压缩
                    message_builder.add_image_content(
                        content_item[0],
                        compress_base64_image_by_scale(content_item[1]),
                    )
                else:
                    message_builder.add_text_content(content_item)
            compressed_messages.append(message_builder.build())
        else:
            compressed_messages.append(message)

    return compressed_messages
