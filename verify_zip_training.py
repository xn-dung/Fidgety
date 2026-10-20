import io, os, subprocess, sys, tempfile, zipfile
from pathlib import Path
import numpy as np
root=Path.cwd()
rng=np.random.default_rng(4)
with tempfile.TemporaryDirectory(dir=root) as directory:
    directory=Path(directory)
    archive=directory/'synthetic.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for modality,time,features in [('mat',500,6),('imu',300,36),('vitpose_kpc',250,30)]:
            prefix='vitpose' if modality=='vitpose_kpc' else modality
            for split,n in [('train',16),('test',4)]:
                for kind,data in [('input',rng.normal(size=(n,time*features))),('output',np.arange(n)%2)]:
                    text=io.StringIO(); np.savetxt(text,data,fmt='%.5f')
                    z.writestr(f'data/data_{modality}_9fold/{kind}_{prefix}_9fold_fold_1_{split}.txt',text.getvalue())
        z.writestr('data/random_indices_9fold/random_indices_train_9fold_fold_1.txt','\n'.join(map(str,range(1,17))))
    env=os.environ.copy()
    env.update(FIDGETY_DATA_ZIP=str(archive),FIDGETY_FOLDS='1',FIDGETY_RUNS='1',FIDGETY_EPOCHS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for script in root.glob('*_conv_best_model_9fold.py'):
        out=directory/script.stem; out.mkdir()
        run=subprocess.run([sys.executable,str(script)],cwd=out,env=env,capture_output=True,text=True)
        if run.returncode: raise RuntimeError(run.stdout+run.stderr)
        assert len(list(out.glob('*.pt')))==1
        result=np.loadtxt(next(out.glob('res_*.txt')))
        assert np.isfinite(result).all()
        print('ZIP -> preprocess -> train -> checkpoint OK:',script.name,flush=True)
