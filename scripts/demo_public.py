"""Data-free smoke demonstration. No actual report text is loaded or printed."""
from pathlib import Path
import json, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from womenhelp_competition.text_augmentation import augment_rows_for_label

def main():
    # Deliberately artificial input; labels exercise code paths, not human judgments.
    rows=[{'record_id':str(i),'text':f'ARTIFICIAL SOFTWARE FIXTURE {i}', 'labels':3 if i<2 else i%3} for i in range(20)]
    _, summary=augment_rows_for_label(rows,target_label_id=3,mode='report_style',target_ratio=.2,max_copies_per_row=2,seed=3407)
    print(json.dumps({'description':'Artificial software check, not a benchmark or a real report','augmentation_counts':summary},indent=2))

if __name__=='__main__':main()
