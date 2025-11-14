# Installation Checklist

## Step 1: Verify Directory Structure
- [ ] D:\RelationalRelativity\ exists
- [ ] D:\RelationalRelativity\qrr_core\ exists
- [ ] D:\RelationalRelativity\rocket_motor_ballistics_upwork\ exists

## Step 2: Copy Core Library
- [ ] Copy enhanced_qrr_core.py to D:\RelationalRelativity\qrr_core\

## Step 3: Copy Project Files
- [ ] Copy proposal document to proposal/
- [ ] Copy simulation code to simulation/
- [ ] Verify rocket_motor_paths.py is in D:\RelationalRelativity\

## Step 4: Test Installation
- [ ] Open PowerShell in simulation/ directory
- [ ] Run: python -c "import sys; sys.path.insert(0, r'D:\RelationalRelativity\qrr_core'); from enhanced_qrr_core import QRRSystem; print('Success!')"

## Step 5: Ready to Run
- [ ] Run simulation: python simulation/qrr_rocket_motor.py
- [ ] Check results appear in results/
- [ ] Verify proposal is ready in proposal/

## Notes

If import errors occur:
1. Check Python path includes D:\RelationalRelativity\qrr_core
2. Verify enhanced_qrr_core.py is in correct location
3. Check Python version (3.8+ required)
