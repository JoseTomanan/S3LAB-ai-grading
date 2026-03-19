<script lang="ts">
  import logo from "$lib/assets/android-chrome-512x512.png";
  import * as Dialog from "$lib/components/ui/dialog/index.ts";

  import IconPaper from '~icons/mdi/paper';
  import IconSection from '~icons/mdi/person';
  import Manual from "./Manual.svelte";
</script>


<div class="container w-full h-[93vh] flex flex-col items-center justify-center gap-y-10">
  <div class="relative w-full py-8 flex flex-col items-center justify-center gap-y-3">
    <span class="absolute inset-0 m-auto size-28 object-contain -z-50 block left-0 right-0 top-0 bottom-0">
      <span class="slow-spin bg-accent/40 rounded-full">
        <img src={logo}
              alt="Llama mascot"
              class="size-28"
          />
      </span>
    </span>
    <h1 class="text-white text-shadow-xs text-4xl w-fit tracking-[-0.08em] leading-7 font-extrabold">
      SIPAT<br>MATH
    </h1>
  </div>
  <div class="space-y-3 w-full max-w-md px-4 sm:px-0">
    <div class="flex flex-col sm:flex-row gap-3 w-full">
      <a href="/instances" class="button-outline flex-1 flex flex-col items-center gap-y-1 py-4 px-4">
        <IconPaper class="size-10 sm:size-12 opacity-80"/>
        <b>Test Instances</b>
        <p class="text-xs text-muted-foreground">Manage exams & grading</p>
      </a>
      <a href="/sections" class="button-outline flex-1 flex flex-col items-center gap-y-1 py-4 px-4">
        <IconSection class="size-10 sm:size-12 opacity-80"/>
        <b>Sections</b>
        <p class="text-xs text-muted-foreground">Manage student rosters</p>
      </a>
    </div>
    <Dialog.Root>
      <Dialog.Trigger class="w-full text-center text-sm text-muted-foreground hover:underline cursor-pointer py-1">
        User manual
      </Dialog.Trigger>
      <Manual />
    </Dialog.Root>
  </div>
</div>
<h6 class="opacity-25">Made for CS199 by Tomanan and Tuan, with love.</h6>
