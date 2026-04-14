import os
import glob
import argparse
import random

def write_csv(video_list, output_csv):
    """動画パスのリストをV-JEPA対応のCSV形式で書き出す"""
    with open(output_csv, 'w') as f:
        for vf in video_list:
            abs_path = os.path.abspath(vf)
            f.write(f"{abs_path} 0\n")
    print(f"Success: Wrote {len(video_list)} paths to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V-JEPA用のデータセットCSVを比率で自動分割生成します")
    parser.add_argument("--data_dir", type=str, required=True, help="動画の入ったルートディレクトリ")
    parser.add_argument("--out_dir", type=str, default="configs/train_2_1/vitg16", help="CSVを出力するディレクトリ")
    parser.add_argument("--ext", type=str, default="*.mp4", help="検索する動画の拡張子")
    parser.add_argument("--ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1], 
                        help="train val eval の分割比率。例: --ratio 0.8 0.1 0.1")
    parser.add_argument("--seed", type=int, default=42, help="ランダムシード値（再現性のため）")
    args = parser.parse_args()

    # 動画ファイルを再帰的に検索
    search_pattern = os.path.join(args.data_dir, "**", args.ext)
    all_videos = glob.glob(search_pattern, recursive=True)
    
    if not all_videos:
        print(f"Warning: No videos found in {args.data_dir} with extension {args.ext}")
        exit()

    # ランダムにシャッフル
    random.seed(args.seed)
    random.shuffle(all_videos)

    total = len(all_videos)
    # 比率に合わせてインデックスを計算
    train_end = int(total * args.ratio[0])
    val_end = train_end + int(total * args.ratio[1])

    # 動画リストを分割
    train_vids = all_videos[:train_end]
    val_vids = all_videos[train_end:val_end]
    eval_vids = all_videos[val_end:]

    print(f"Total videos found: {total}")
    print(f"Splitting into Train: {len(train_vids)}, Val: {len(val_vids)}, Eval: {len(eval_vids)}")

    # それぞれCSVに出力
    os.makedirs(args.out_dir, exist_ok=True)
    if train_vids: write_csv(train_vids, os.path.join(args.out_dir, "autodriving_train_paths.csv"))
    if val_vids: write_csv(val_vids, os.path.join(args.out_dir, "autodriving_val_paths.csv"))
    if eval_vids: write_csv(eval_vids, os.path.join(args.out_dir, "autodriving_eval_paths.csv"))
