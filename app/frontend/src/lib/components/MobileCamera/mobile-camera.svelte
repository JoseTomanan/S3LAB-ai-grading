<script lang="ts">
  /**
   * Modified @cloudparker/easy-camera-svelte to fit mobile environment.
   * Source code credits go to them.
   */
  type Props = {
    width?: number;
    style?: string;
    autoOpen?: boolean;
    mirrorDisplay?: boolean;
    useFrontCamera?: boolean;
    useAudio?: boolean;
    needFrames?: boolean;
    onVideo?: (blob: Blob) => void;
    onStartVideoRecording?: () => void;
    onStopVideoRecording?: () => void;
    onImage?: (imageDate: ImageData | string) => void;
    onOpen?: () => void;
    onClose?: () => void;
    onPause?: () => void;
    onResume?: () => void;
    onInit?: (devices: MediaDeviceInfo[]) => void;
    onFrame?: (imageData: ImageData) => void;
  };

  let {
    width = $bindable(340),
    style = '',
    autoOpen = true,
    mirrorDisplay = $bindable(true),
    useFrontCamera = true,
    useAudio = true,
    needFrames = false,
    onVideo,
    onStartVideoRecording,
    onStopVideoRecording,
    onImage,
    onOpen,
    onClose,
    onPause,
    onResume,
    onInit,
    onFrame
  }: Props = $props();

  let isOpened: boolean;
  let videoRef: HTMLVideoElement | null = $state(null);
  let canvasRef: HTMLCanvasElement;
  let stream: MediaStream | null;
  let ctx: CanvasRenderingContext2D;
  let mediaRecorder: MediaRecorder | null;
  let recordedChunks: Blob[] = [];
  let cameras: MediaDeviceInfo[];
  let frontCamera: MediaDeviceInfo;
  let rearCamera: MediaDeviceInfo;
  let isPause: boolean = false;

  let ratio: number = $derived.by(() => {
    if (videoRef) {
      return videoRef.videoHeight / videoRef.videoWidth;
    }
    return 1;
  });

  let height: number = $derived.by(() => {
    if (ratio && videoRef) {
      width = width || videoRef.videoWidth;
      return width * ratio;
    }
    return width;
  });

  // let dispatch = createEventDispatcher();

  export const startVideoRecording = (): Promise<Blob | null> => {
    return new Promise((resolve, reject) => {
      if (stream) {
        recordedChunks = [];
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
          recordedChunks.push(event.data);
        };
        mediaRecorder.onstop = () => {
          const blob = new Blob(recordedChunks, { type: 'video/mp4' });
          onVideo && onVideo(blob);
          resolve(blob);
          mediaRecorder = null;
          recordedChunks = [];
        };

        mediaRecorder.start();
        onStartVideoRecording && onStartVideoRecording();
      }
    });
  };

  export const stopVideoRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
      onStopVideoRecording && onStopVideoRecording();
    }
  };

  export const captureImage = async (isImageData?: boolean): Promise<ImageData | string | null> => {
    let imageData: any = null;
    if (ctx && canvasRef && isOpened) {
      if (isImageData) {
        imageData = ctx.getImageData(0, 0, width, height);
      } else {
        imageData = canvasRef.toDataURL();
      }

      onImage && onImage(imageData);
      return imageData;
    }
    return imageData;
  };

  export const open = (): void => {
    if (canvasRef) {
      openCamera();
    }
  };

  export const close = (): void => {
    if (canvasRef) {
      closeCamera();
    }
  };

  export const pause = (): void => {
    isPause = true;
    if (mediaRecorder) {
      mediaRecorder.pause();
    }
    handleResume();
  };

  export const resume = (): void => {
    isPause = false;
    if (mediaRecorder) {
      mediaRecorder.resume();
    }
    handlePause();
  };

  export const isPlaying = (): boolean => {
    return isOpened;
  };

  export const isRecording = (): boolean => {
    return !!mediaRecorder;
  };

  export const switchCamera = (deviceId?: string): void => {
    useFrontCamera = !useFrontCamera;
    openCamera(deviceId);
  };

  export const getCameraDevices = async (): Promise<MediaDeviceInfo[]> => {
    let devices: MediaDeviceInfo[] = await navigator.mediaDevices.enumerateDevices();
    const cameras: MediaDeviceInfo[] = [];
    (devices || []).forEach((device: MediaDeviceInfo) => {
      if (device.kind === 'videoinput') {
        cameras.push(device);
        if (device.label.includes('front')) {
          frontCamera = device;
        } else {
          rearCamera = device;
        }
      }
    });
    return cameras;
  };

  const handleOpen = () => {
    isOpened = true;
    onOpen && onOpen();
  };

  const handleClose = () => {
    onClose && onClose();
  };

  const handlePause = () => {
    onPause && onPause();
  };

  const handleResume = () => {
    onResume && onResume();
  };

  const initCamera = async () => {
    if (videoRef && canvasRef) {
      ctx = canvasRef.getContext('2d', { willReadFrequently: true })!;
      videoRef.addEventListener('canplay', () => {
        // ratio = videoRef.videoHeight / videoRef.videoWidth;
        // adjustResolution(width);
        // ctx.drawImage(videoRef, 0, 0, width, height);
        handleOpen();
        drawCanvas();
      });
      if (autoOpen) {
        openCamera();
      }
      let devices = await getCameraDevices();
      if (devices) {
        onInit && onInit(devices);
      }
    }
  };

  const openCamera = (deviceId?: string) => {
    if (isOpened) {
      closeCamera();
    }

    let constraints: any = { video: true, audio: useAudio };

    if (deviceId) {
      constraints = {
        video: {
          deviceId: deviceId
        },
        audio: useAudio
      };
    } else {
      if (useFrontCamera && frontCamera) {
        constraints = {
          video: {
            deviceId: frontCamera.deviceId
          },
          audio: useAudio
        };
      } else if (rearCamera) {
        constraints = {
          video: {
            deviceId: rearCamera.deviceId
          },
          audio: useAudio
        };
      }
    }

    navigator.mediaDevices
      .getUserMedia(constraints)
      .then((mediaStream) => {
        if (videoRef) {
          videoRef.srcObject = mediaStream;
          stream = mediaStream;
          videoRef.play();
        }
      })
      .catch((error) => {
        console.error('Unable to access the camera.', error);
      });
  };

  const closeCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => {
        track.stop();
      });
      handleClose();
      if (mediaRecorder) {
        mediaRecorder.stop();
        mediaRecorder = null;
      }
      stream = null;
    }
  };

  const drawCanvas = () => {
    if (videoRef && canvasRef && width && height && isOpened) {
      if (!isPause) {
        if (stream) {
          if (mirrorDisplay) {
            // flip image | mirror image
            ctx.scale(-1, 1);
            ctx.drawImage(videoRef, -width, 0, width, height);
            ctx.scale(-1, 1);
          } else {
            ctx.drawImage(videoRef, 0, 0, width, height);
          }
          if (needFrames) {
            onFrame && onFrame(ctx.getImageData(0, 0, width, height));
          }
        }
      }
    }

    requestAnimationFrame(drawCanvas);
  };

  $effect(() => {
    initCamera();
    return () => {
      closeCamera();
    };
  });
</script>


<div class="qrcode-sanner-container">
  <video id="camera-video"
          class="camera-video"
          bind:this={videoRef}
          {width} {height} muted playsinline>
    <track kind="captions" />
  </video>
  <canvas id="camera-canvas"
            class="camera-canvas"
            style={style}
            bind:this={canvasRef} {width} {height}
  ></canvas>
</div>


<style>
  .qrcode-sanner-container {
    width: 100%;
    height: 100%;
  }

  .camera-video {
    display: none;
  }

  .camera-canvas {
    width: 100%;
    height: 100%;
    object-fit: contain; /* preserves aspect ratio */
    display: block;
  }
</style>