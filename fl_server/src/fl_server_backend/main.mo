import Array "mo:base/Array";
import Float "mo:base/Float";

persistent actor {

  var localLoss1 : [Float] = [];
  var localLoss2 : [Float] = [];
  var localLoss3 : [Float] = [];
  var localLoss4 : [Float] = [];

  var pAug1_c1 : [Float] = []; var pAug2_c1 : [Float] = [];
  var pAug1_c2 : [Float] = []; var pAug2_c2 : [Float] = [];
  var pAug1_c3 : [Float] = []; var pAug2_c3 : [Float] = [];
  var pAug1_c4 : [Float] = []; var pAug2_c4 : [Float] = [];

  private func kl_div(p_clean : [Float], p_aug : [Float]) : Float {
    
    var kl : Float = 0.0;
    let n = p_clean.size();
    var i = 0;
    while (i < n) {
      if (p_clean < 0.0 or p_aug < 0.0) {
        return 999999.0; // Massive KL divergence = 0% voting weight
      };
      let p = if (p_clean[i] < 1e-7) 1e-7 else p_clean[i];
      let q = if (p_aug[i]  < 1e-7) 1e-7 else p_aug[i];
      kl += q * (Float.log(q) - Float.log(p));
      i += 1;
    };
    kl
  };

  public func client1(local_loss : [Float], p_aug1 : [Float], p_aug2 : [Float]) : async () {
    localLoss1 := local_loss; pAug1_c1 := p_aug1; pAug2_c1 := p_aug2;
  };
  public func client2(local_loss : [Float], p_aug1 : [Float], p_aug2 : [Float]) : async () {
    localLoss2 := local_loss; pAug1_c2 := p_aug1; pAug2_c2 := p_aug2;
  };
  public func client3(local_loss : [Float], p_aug1 : [Float], p_aug2 : [Float]) : async () {
    localLoss3 := local_loss; pAug1_c3 := p_aug1; pAug2_c3 := p_aug2;
  };
  public func client4(local_loss : [Float], p_aug1 : [Float], p_aug2 : [Float]) : async () {
    localLoss4 := local_loss; pAug1_c4 := p_aug1; pAug2_c4 := p_aug2;
  };

  public query func testAverage() : async [Float] {
    let size = localLoss1.size();
    if (size == 0) { return []; };

    let kl1 = kl_div(localLoss1, pAug1_c1) + kl_div(localLoss1, pAug2_c1);
    let kl2 = kl_div(localLoss2, pAug1_c2) + kl_div(localLoss2, pAug2_c2);
    let kl3 = kl_div(localLoss3, pAug1_c3) + kl_div(localLoss3, pAug2_c3);
    let kl4 = kl_div(localLoss4, pAug1_c4) + kl_div(localLoss4, pAug2_c4);

    let trust1 = 1.0 / (kl1 + 1e-7);
    let trust2 = 1.0 / (kl2 + 1e-7);
    let trust3 = 1.0 / (kl3 + 1e-7);
    let trust4 = 1.0 / (kl4 + 1e-7);

    let kl_sum = trust1 + trust2 + trust3 + trust4 + 1e-7;

    let w1 = trust1 / kl_sum;
    let w2 = trust2 / kl_sum;
    let w3 = trust3 / kl_sum;
    let w4 = trust4 / kl_sum;

    Array.tabulate<Float>(size, func(i : Nat) : Float {
      localLoss1[i] * w1 + localLoss2[i] * w2 +
      localLoss3[i] * w3 + localLoss4[i] * w4
    });
  };
};