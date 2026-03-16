<script lang="ts">
  import logo from "$lib/assets/android-chrome-512x512.png";
	import * as Dialog from "$lib/components/ui/dialog/index.ts";

  import IconPaper from '~icons/mdi/paper';
  import IconSection from '~icons/mdi/person';
	import Manual from "./Manual.svelte";
</script>


<div class="container w-full h-[93vh] flex flex-col items-center justify-center gap-y-16">
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
  <div class="space-y-4">
    <div class="flex flex-row gap-x-4 w-full justify-center
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
    <Dialog.Root>
      <Dialog.Trigger class="button-outline w-full">
        User manual
      </Dialog.Trigger>
      <Manual />
    </Dialog.Root>
  </div>
</div>
<h6 class="opacity-40">Made for CS199 by Tomanan and Tuan, with love.</h6>
