"""Instagram Automation & API Connector Script (Micro-AGI)."""

import os
import json
import urllib.request
import urllib.parse

class InstagramConnector:
    """Interacts with Meta Graph API for Instagram Business/Creator accounts."""

    def __init__(self, access_token: str = None, instagram_account_id: str = None):
        self.access_token = access_token or os.getenv("INSTAGRAM_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")
        self.account_id = instagram_account_id or os.getenv("INSTAGRAM_ACCOUNT_ID", "YOUR_ACCOUNT_ID")
        self.graph_base = "https://graph.facebook.com/v19.0"

    def get_profile_info(self) -> dict:
        """Fetches basic profile information."""
        url = f"{self.graph_base}/{self.account_id}?fields=username,name,biography,followers_count,media_count&access_token={self.access_token}"
        try:
            req = urllib.request.urlopen(url, timeout=10)
            return json.loads(req.read().decode("utf-8"))
        except Exception as e:
            return {"status": "error", "message": str(e), "hint": "Provide a valid Meta Graph API Token"}

    def publish_photo(self, image_url: str, caption: str) -> dict:
        """Two-step publishing container creation and media publish."""
        container_url = f"{self.graph_base}/{self.account_id}/media"
        data = urllib.parse.urlencode({"image_url": image_url, "caption": caption, "access_token": self.access_token}).encode()
        try:
            req = urllib.request.Request(container_url, data=data, method="POST")
            resp = urllib.request.urlopen(req, timeout=10)
            res_json = json.loads(resp.read().decode("utf-8"))
            creation_id = res_json.get("id")
            
            # Publish container
            publish_url = f"{self.graph_base}/{self.account_id}/media_publish"
            pdata = urllib.parse.urlencode({"creation_id": creation_id, "access_token": self.access_token}).encode()
            preq = urllib.request.Request(publish_url, data=pdata, method="POST")
            return json.loads(urllib.request.urlopen(preq).read().decode("utf-8"))
        except Exception as e:
            return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    client = InstagramConnector()
    print("Instagram Client initialized. Ready for API token configuration.")
