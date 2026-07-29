from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web

from harness import check_speculative_parity as parity


class ParityHarnessTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        async def completion(request: web.Request) -> web.Response:
            body = await request.json()
            seed = json.dumps(
                body["messages"], ensure_ascii=False, sort_keys=True
            )
            content = f"deterministic:{parity.digest(seed)}"
            return web.json_response(
                {"choices": [{"message": {"content": content}}]}
            )

        self.app = web.Application()
        self.app.router.add_post("/v1/chat/completions", completion)
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        socket = self.site._server.sockets[0]
        self.base_url = f"http://127.0.0.1:{socket.getsockname()[1]}"
        self.temp_dir = tempfile.TemporaryDirectory()

    async def asyncTearDown(self) -> None:
        await self.runner.cleanup()
        self.temp_dir.cleanup()

    async def test_record_then_compare(self) -> None:
        reference = Path(self.temp_dir.name) / "reference.json"
        args = SimpleNamespace(
            base_url=self.base_url,
            model="test-model",
            max_tokens=8,
            timeout=5.0,
            reference=reference,
        )

        await parity.record(args)
        self.assertTrue(reference.is_file())
        payload = json.loads(reference.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["records"]), 16)

        await parity.compare(args)


if __name__ == "__main__":
    unittest.main()
