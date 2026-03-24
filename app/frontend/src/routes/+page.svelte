<script lang="ts">
  import logo from "$lib/assets/android-chrome-512x512.png";
  import * as Dialog from "$lib/components/ui/dialog/index.ts";

  import toast from "svelte-5-french-toast";
  
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
    <div
      class="flex flex-col sm:flex-row gap-3 w-full
              [&>a]:flex-1 [&>a]:flex [&>a]:flex-col
              [&>a]:items-center [&>a]:gap-y-1
              [&>a]:py-4 [&>a]:px-4
              [&>a>svg]:size-10 [&>a>svg]:sm:size-12 [&>a>svg]:opacity-80
              [&>a>b]:font-bold [&>a>p]:text-xs [&>a>p]:text-muted-foreground
              ">
      <a href="/instances" class="button-outline">
        <IconPaper />
        <b>Test Instances</b>
        <p>Manage exams & grading</p>
      </a>
      <a href="/sections" class="button-outline">
        <IconSection />
        <b>Sections</b>
        <p>Manage student rosters</p>
      </a>
    </div>
    <Dialog.Root>
      <Dialog.Trigger class="w-full text-center text-sm text-muted-foreground hover:underline cursor-pointer py-1"
                      onclick={() => {toast("Coming soon!", {icon: "🚧"})}}>
        User manual
      </Dialog.Trigger>
      <!-- <Manual /> -->
    </Dialog.Root>
  </div>
</div>
<h6 class="opacity-25">Made for CS199 by Tomanan and Tuan, with love.</h6>
