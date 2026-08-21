import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "download_sensenova_u15_int8.ps1"
BATCH = ROOT / "download_sensenova_u15_int8.bat"


class SenseNovaDownloadScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = SCRIPT.read_text(encoding="utf-8")
        cls.batch = BATCH.read_text(encoding="utf-8")

    def test_model_download_is_immutable_and_integrity_checked(self):
        self.assertIn(
            'ModelRevision = "e63b0a7e483bffdb1ff0463a39fbfd04ad3c85d9"', self.script
        )
        self.assertIn("ModelBytes = [Int64]19947887936", self.script)
        self.assertIn(
            'ModelSha256 = "8b655046f6e22c22258607556cacee3c1d82ae534146fb9c0faba04a0e4b3c8f"',
            self.script,
        )
        self.assertIn("Get-FileHash -LiteralPath $Path -Algorithm SHA256", self.script)
        self.assertIn('if ($magic -ne "GGUF"', self.script)

    def test_runtime_source_is_pinned_and_blob_verified(self):
        self.assertIn(
            'SourceRevision = "12a2bd9cba22a5317164b55db4f7c6209a371f83"', self.script
        )
        self.assertIn("Get-GitBlobSha1", self.script)
        self.assertIn('$_.path -like "src/sensenova_u1/*"', self.script)
        self.assertIn("sentencepiece==0.2.1", self.script)

    def test_download_is_resumable_and_batch_calls_the_script(self):
        self.assertIn("--continue-at -", self.script)
        self.assertIn("--retry-all-errors", self.script)
        self.assertIn("--parallel-max 8", self.script)
        self.assertIn('("chunk-{0:D3}.bin" -f $index)', self.script)
        self.assertIn("$input.CopyTo($output)", self.script)
        self.assertIn("download_sensenova_u15_int8.ps1", self.batch)


if __name__ == "__main__":
    unittest.main()
