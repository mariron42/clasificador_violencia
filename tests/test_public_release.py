from pathlib import Path
import csv, tempfile, sys, unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from womenhelp_competition.text_augmentation import augment_rows_for_label
from womenhelp_competition.data import load_subtask1, load_subtask2
from womenhelp_competition.labels import SUBTASK1_LABELS, SUBTASK2_LABELS
from womenhelp_competition.metrics import compute_multiclass_metrics

class PublicReleaseTests(unittest.TestCase):
    def test_augmentation_is_deterministic_bounded_and_preserves_originals(self):
        rows=[{'record_id':str(i),'text':f'ARTIFICIAL FIXTURE {i}','labels':3 if i<2 else 0,'type_labels':[0.0]*7} for i in range(20)]
        args=dict(target_label_id=3,mode='report_style',target_ratio=.2,max_copies_per_row=2,seed=3407)
        result,summary=augment_rows_for_label(rows,**args)
        self.assertEqual((result,summary),augment_rows_for_label(rows,**args))
        self.assertEqual(result[:len(rows)],rows)
        self.assertEqual(summary['added_rows'],3)
        self.assertTrue(all(r['labels']==3 and r['type_labels']==[0.0]*7 for r in result[20:]))
        self.assertEqual(len({r['record_id'] for r in result}),len(result))
        self.assertLessEqual(summary['added_rows'],4)

    def test_missing_target_class_creates_no_rows(self):
        rows=[{'record_id':'1','text':'ARTIFICIAL FIXTURE','labels':0}]
        result,summary=augment_rows_for_label(rows,target_label_id=3,mode='report_style',target_ratio=.2,max_copies_per_row=2,seed=1)
        self.assertEqual(result,rows);self.assertEqual(summary['added_rows'],0)

    def test_official_column_shapes_load_from_explicit_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            for sub in ['subtask1','subtask2']:(p/sub).mkdir()
            with (p/'subtask1/train.csv').open('w',newline='',encoding='utf8') as f:
                csv.writer(f).writerows([['ID','TEXT','CLASS'],['1','ARTIFICIAL FIXTURE','3']])
            with (p/'subtask2/train.csv').open('w',newline='',encoding='utf8') as f:
                csv.writer(f).writerows([['ID','Text',*[f'L{i}' for i in range(7)]],['1','ARTIFICIAL FIXTURE',0,1,0,1,0,0,0]])
            self.assertEqual(load_subtask1('train',p)[0].severity_id,'3')
            self.assertEqual(load_subtask2('train',p)[0].label_vector['L3'],1)

    def test_metric_class_order_and_hand_calculated_case(self):
        self.assertEqual(SUBTASK1_LABELS,['Mild','Medium','High','Severe'])
        self.assertEqual(len(SUBTASK2_LABELS),7)
        m=compute_multiclass_metrics(['Mild','Mild','Severe','Severe'],['Mild','Severe','Severe','Severe'],['Mild','Severe'])
        self.assertAlmostEqual(m['accuracy'],.75)
        self.assertAlmostEqual(m['macro_f1'],(2/3+.8)/2)

if __name__=='__main__':unittest.main()
