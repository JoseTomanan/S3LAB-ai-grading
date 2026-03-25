import type { FileRecord } from "$lib/types/ui.ts";



export async function rotateImage(record: FileRecord, isCw: boolean): Promise<FileRecord> {
  const degrees = isCw ? 90 : 270;

  const bitmap = await createImageBitmap(record.file);
  const { width, height } = bitmap;

  const swap = degrees === 90 || degrees === 270;
  const canvas = document.createElement("canvas");
  canvas.width = swap ? height : width;
  canvas.height = swap ? width : height;

  const ctx = canvas.getContext("2d")!;
  ctx.translate(canvas.width / 2, canvas.height / 2);
  ctx.rotate((degrees * Math.PI) / 180);
  ctx.drawImage(bitmap, -width / 2, -height / 2);
  bitmap.close();

  const blob = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error("toBlob failed"))),
      record.file.type || "image/jpeg"
    )
  );

  URL.revokeObjectURL(record.url);

  return {
    file: new File([blob], record.file.name, {
      type: blob.type,
      lastModified: Date.now(),
    }),
    name: record.name,
    url: URL.createObjectURL(blob),
    statusCode: record.statusCode,
  };
}


export async function flipImage(record: FileRecord, isFlipHorizontal: boolean): Promise<FileRecord> {
  const bitmap = await createImageBitmap(record.file);
  const { width, height } = bitmap;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d")!;

  if (isFlipHorizontal) {
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
  } else {
    ctx.translate(0, height);
    ctx.scale(1, -1);
  }

  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();

  const blob = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error("toBlob failed"))),
      record.file.type || "image/jpeg"
    )
  );

  URL.revokeObjectURL(record.url);

  return {
    file: new File([blob], record.file.name, {
      type: blob.type,
      lastModified: Date.now(),
    }),
    name: record.name,
    url: URL.createObjectURL(blob),
    statusCode: record.statusCode,
  };
}


async function handleRotateCommand(singleRecord: FileRecord, records: FileRecord[], isCw: boolean) {
  if (!singleRecord)
    return;
  const recordResponse = await rotateImage(singleRecord, isCw);
  records = records.map(i => i.name == recordResponse.name ? recordResponse : i);
}


async function handleFlipCommand(singleRecord: FileRecord, records: FileRecord[], isFlipHorizontally: boolean) {
  if (!singleRecord)
    return;
  const recordResponse = await flipImage(singleRecord, isFlipHorizontally);
  records = records.map(i => i.name == recordResponse.name ? recordResponse : i);
}
