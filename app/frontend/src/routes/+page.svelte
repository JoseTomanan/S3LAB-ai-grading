<script lang="ts">
  import logo from "$lib/assets/android-chrome-512x512.png";

  import IconPaper from '~icons/mdi/paper';
  import IconSection from '~icons/mdi/person';
</script>


<div class="container w-full h-[95vh] flex flex-col items-center justify-center gap-y-16">
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
  <div class="flex flex-row gap-x-1.5 w-full justify-evenly
          [&>a]:text-center [&>a]:px-2 [&>a>*]:opacity-80">
    <a href="/instances" class="button-primary">
      <IconPaper class="size-28"/>
      <b>Test Instances</b>
    </a>
    <a href="/sections" class="button-primary">
      <IconSection class="size-28"/>
      <b>Sections</b>
    </a>
  </div>
  <!-- <button class="button-outline">
    Guide
  </button> -->
</div>
<h6 class="opacity-40">Made for CS199 by Tomanan and Tuan, with love.</h6>
