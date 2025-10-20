import requests


if __name__ == "__main__":
	with open("dataset/DrawEduMath_QA_ImageURL.txt", "r") as f:
		links = [line.strip() for line in f if line.strip()]

	for i, url in enumerate(links, 1):
		try:
			response = requests.get(url, timeout=10)
			response.raise_for_status()
			ext = url.split(".")[-1].split("?")[0]
			with open(f"image_{i}.{ext}", "wb") as f_out:
				f_out.write(response.content)
			print(f"Downloaded {i}/{len(links)}")
		except Exception as e:
			print(f"Failed {url}: {e}")