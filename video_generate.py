import argparse
import os
import time
import random

from diffusers.utils import export_to_video
from diffusers.utils import load_image

from skyreelsinfer import TaskType
from skyreelsinfer.offload import OffloadConfig
from skyreelsinfer.skyreels_video_infer import SkyReelsVideoInfer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V1-Hunyuan-T2V")
    parser.add_argument("--outdir", type=str, default="skyreels")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_frames", type=int, default=97)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default="FPS-24, A 3D model of a 1800s victorian house.")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
    )
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--video_num", type=int, default=2)
    parser.add_argument("--task_type", type=str, default="t2v", choices=["t2v", "i2v"])
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--embedded_guidance_scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quant", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--high_cpu_memory", action="store_true")
    parser.add_argument("--parameters_level", action="store_true")
    parser.add_argument("--compiler_transformer", action="store_true")
    parser.add_argument("--sequence_batch", action="store_true")
    parser.add_argument("--mbps", type=float, default=7)

    args = parser.parse_args()

    out_dir = f"results/{args.outdir}"
    os.makedirs(out_dir, exist_ok=True)

    if args.task_type == "i2v":
        image = load_image(args.image)
    
    if args.seed == -1:
        random.seed(time.time())
        args.seed = int(random.randrange(4294967294))

    predictor = SkyReelsVideoInfer(
        task_type=TaskType.I2V if args.task_type == "i2v" else TaskType.T2V,
        model_id=args.model_id,
        quant_model=args.quant,
        world_size=args.gpu_num,
        is_offload=args.offload,
        offload_config=OffloadConfig(
            high_cpu_memory=args.high_cpu_memory,
            parameters_level=args.parameters_level,
            compiler_transformer=args.compiler_transformer,
        ),
        enable_cfg_parallel=args.guidance_scale > 1.0,
    )
    print("finish pipeline init")
    kwargs = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "embedded_guidance_scale": args.embedded_guidance_scale,
        "negative_prompt": args.negative_prompt,
        "cfg_for": args.sequence_batch,
    }
    if args.task_type == "i2v":
        kwargs["image"] = image


    #20250223 pftq: customizable bitrate
    def save_video_with_quality(frames, output_path, fps, bitrate):
        import cv2
        import numpy as np
        frames = [np.array(frame) for frame in frames]
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        writer.set(cv2.CAP_PROP_BITRATE, bitrate)
        for frame in frames:
            writer.write(frame)
        writer.release()
    
    for idx in range(args.video_num):
        output = predictor.inference(kwargs)
        #video_out_file = f"{args.prompt[:100].replace('/','')}_{args.seed}_{idx}.mp4"
        #export_to_video(output, f"{out_dir}/{video_out_file}", fps=args.fps)

        #20250223 pftq: More useful filename and higher customizable bitrate
        from datetime import datetime
        now = datetime.now()
        formatted_time = now.strftime('%Y-%m-%d_%H-%M-%S')
        video_out_file = formatted_time+f"_cfg-{args.guidance_scale}_steps-{args.num_inference_steps}_{args.prompt[:20].replace('/','')}_{idx}.mp4"
        bitrate_bps = int(args.mbps * 1000)
        save_video_with_quality(output, f"{out_dir}/{video_out_file}", args.fps, bitrate_bps)
