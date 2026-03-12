<script lang="ts">
  import logo from "$lib/assets/android-chrome-512x512.png";

  import IconLink from '~icons/mdi/arrow-top-right';
</script>


<div class="container w-full h-9/10 flex flex-col items-center justify-center gap-y-6">
  <div class="relative w-full py-10 flex items-center justify-center">
    <span class="absolute inset-0 m-auto size-32 object-contain -z-50 block left-0 right-0 top-0 bottom-0">
      <span class="slow-spin bg-accent/40 rounded-full">
        <img src={logo}
              alt="Llama mascot"
              class="size-32"
          />
      </span>
    </span>
    <h1 class="text-white text-shadow-xs text-4xl w-fit tracking-[-0.08em] leading-7 font-extrabold">
      SIPAT<br>MATH
    </h1>
  </div>
  <div class="flex flex-col gap-y-1.5 w-full
          [&>a]:text-center">
    <a href="/instances" class="button-primary">
      Test instances
    </a>
    <a href="/sections" class="button-outline">
      Sections
    </a>
    <a href="https://github.com/JoseTomanan/S3LAB-ai-grading"
          class="button-outline flex flex-row items-center justify-center gap-x-1 hover:gap-x-3 transition-all">
      Git Repository
      <IconLink />
    </a>
  </div>
</div>
<h6 class="opacity-40">Made for CS199 by Tomanan and Tuan, with love.</h6>
