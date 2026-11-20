"""Small mathematical and end-to-end checks for the exported estimator."""
import unittest
import torch
from brot import BROTConfig,BROTTrainer
from brot.ot.kantorovich import solve_kantorovich
from brot.ot.barycentric import barycentric_projection


class CoreTests(unittest.TestCase):
    def test_uniform_coupling_and_barycentric_identity(self):
        x=torch.tensor([[0.,0.],[1.,0.],[0.,2.]])
        coupling=solve_kantorovich(x,x)
        self.assertTrue(torch.allclose(coupling.sum(0),torch.full((3,),1/3)))
        self.assertTrue(torch.allclose(coupling.sum(1),torch.full((3,),1/3)))
        self.assertTrue(torch.allclose(barycentric_projection(coupling,x),x,atol=1e-6))

    def test_training_learns_barycentric_targets(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        x=torch.randn(64,2)
        y=1.2*x+torch.tensor([1.,-1.])
        trainer=BROTTrainer(BROTConfig(in_dim=2,out_dim=2,hidden_dims=[16,16],
                                      n_epochs=50,batch_size=64,lr=0.01,
                                      alpha=0.,verbose=False))
        _,targets=trainer.compute_coupling(x,y)
        initial=float((trainer.transform(x)-targets).square().mean())
        trainer.fit(x,y,Y_tilde=targets)
        pred=trainer.transform(x)
        self.assertEqual(pred.shape,x.shape)
        self.assertTrue(torch.isfinite(pred).all())
        self.assertLess(float((pred-targets).square().mean()),initial*0.25)


if __name__=="__main__":
    unittest.main()
