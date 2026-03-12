"""
gradcam viz tool
shows what model looks at
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import os
import pandas as pd

mp='experiments/FER_Baseline_20260202_191825/best_model.h5'
csv=r'C:\Users\dsdjs\Documents\canada\project\paz\paz\datasets\FER\fer2013.csv'
s=48
n=7
ems=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
od='gradcam_results'
os.makedirs(od,exist_ok=True)

# make heatmap
def heat(ia,m,ln,pi=None):
    gm=Model(inputs=m.input,outputs=[m.get_layer(ln).output,m.output])
    
    with tf.GradientTape() as t:
        co,pr=gm(ia)
        if pi is None:pi=tf.argmax(pr[0])
        cc=pr[:,pi]
    
    gr=t.gradient(cc,co)
    pg=tf.reduce_mean(gr,axis=(0,1,2))
    co=co[0]
    pg=pg[:,tf.newaxis,tf.newaxis]
    co=co*pg
    h=tf.reduce_sum(co,axis=-1)
    h=tf.maximum(h,0)
    mh=tf.reduce_max(h)
    
    if mh==0:h=tf.ones_like(h)*0.5
    else:h=h/mh
    
    hn=h.numpy()
    if len(hn.shape)==0:hn=np.ones((6,6))*0.5
    elif len(hn.shape)==1:
        sz=int(np.sqrt(hn.shape[0]))
        hn=hn[:sz*sz].reshape(sz,sz)
    return hn

# overlay heatmap on image
def over(im,h,a=0.4):
    if not isinstance(h,np.ndarray):h=np.array(h)
    if not isinstance(im,np.ndarray):im=np.array(im)
    if len(h.shape)!=2:h=np.ones((im.shape[0],im.shape[1]))*0.5
    
    hr=cv2.resize(h,(im.shape[1],im.shape[0]))
    hc=cv2.applyColorMap(np.uint8(255*hr),cv2.COLORMAP_JET)
    
    if len(im.shape)==2:ir=cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    else:ir=im
    if ir.max()<=1.0:ir=np.uint8(255*ir)
    
    return cv2.addWeighted(hc,a,ir,1-a,0)

# main function
def gen(m,ns=7):
    print("generating gradcam")
    
    # find last conv
    ln=None
    for l in reversed(m.layers):
        if 'conv' in l.name.lower():
            ln=l.name
            break
    if ln is None:
        print("no conv layer")
        return
    print(f"using {ln}")
    
    # load data
    d=pd.read_csv(csv)
    td=d[d['Usage']!='Training']
    
    # get samples
    spe={}
    for ei in range(n):
        es=td[td['emotion']==ei]
        if len(es)>0:
            sp=es.iloc[0]
            px=np.array(sp['pixels'].split(),dtype=np.uint8)
            im=px.reshape(s,s)
            imn=im.astype('float32')/255.0
            imi=imn.reshape(1,s,s,1)
            spe[ei]={'image':im,'image_input':imi,'true_label':ei}
    
    print(f"loaded {len(spe)} samples")
    
    # gen gradcam
    res=[]
    for ei,sd in spe.items():
        print(f"processing {ems[ei]}")
        pr=m.predict(sd['image_input'],verbose=0)
        pc=np.argmax(pr[0])
        cf=pr[0][pc]*100
        
        # get heatmap
        hm=heat(sd['image_input'],m,ln,pred_index=pc)
        ov=over(sd['image'],hm,alpha=0.4)
        
        res.append({
            'emotion':ems[ei],
            'image':sd['image'],
            'heatmap':hm,
            'overlay':ov,
            'predicted':ems[pc],
            'confidence':cf
        })
    
    # make grid
    print("making grid")
    ne=len(res)
    fig,ax=plt.subplots(ne,3,figsize=(12,ne*3))
    if ne==1:ax=ax.reshape(1,-1)
    
    for i,r in enumerate(res):
        ax[i,0].imshow(r['image'],cmap='gray')
        ax[i,0].set_title(f"original: {r['emotion']}")
        ax[i,0].axis('off')
        
        ax[i,1].imshow(r['heatmap'],cmap='jet')
        ax[i,1].set_title("heatmap")
        ax[i,1].axis('off')
        
        ax[i,2].imshow(r['overlay'])
        ax[i,2].set_title(f"predicted: {r['predicted']} ({r['confidence']:.1f}%)")
        ax[i,2].axis('off')
    
    plt.suptitle('gradcam viz\nred=important blue=not',fontsize=14,fontweight='bold',y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(od,'gradcam_all.png'),dpi=300,bbox_inches='tight')
    plt.close()
    
    # save individual
    for r in res:
        en=r['emotion'].lower()
        cv2.imwrite(os.path.join(od,f'{en}_gc.png'),cv2.cvtColor(r['overlay'],cv2.COLOR_RGB2BGR))
    
    # report
    with open(os.path.join(od,'report.txt'),'w') as f:
        f.write("GradCAM Analysis\n\n")
        f.write("shows what model focuses on\n")
        f.write("red=high importance blue=low\n\n")
        f.write("Results:\n")
        for r in res:
            f.write(f"{r['emotion']}: predicted {r['predicted']} ({r['confidence']:.1f}%)\n")
        f.write("\nExpected patterns:\n")
        f.write("happy->mouth, surprise->eyes, angry->eyebrows\n")
        f.write("sad->mouth corners, fear->eyes, disgust->nose\n")
    
    print("done")

if __name__=="__main__":
    print("loading model")
    m=load_model(mp)
    print("loaded")
    gen(m,num_samples=7)
    print("check gradcam_results folder")