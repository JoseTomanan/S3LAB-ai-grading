<script lang="ts">
  const { data, children } = $props();
  import { buttonVariants } from '$lib/components/ui/button/index.ts';
  import IconBack from "~icons/mdi/chevron-left";
  import IconSectionList from '~icons/mdi/format-list-bulleted';

  const sectionId = $derived(Number(data.section_id));
  const sectionName = $derived(data.sections.find(
    (section) => section.section_id === sectionId
  )!.section_name);
</script>


<nav class="bg-sidebar text-sidebar-foreground
            shadow shadow-sidebar-border
            w-full flex flex-row items-center justify-between p-4 pt-6">
  <a href="/sections" class={buttonVariants({ variant: 'floating' })}>
    <IconBack class="size-8"/>
  </a>
  <span class="*:leading-5 *:text-center">
    <h2>{sectionName}</h2>
  </span>
  <a href="/instances" class={buttonVariants({ variant: 'floating' })}>
    <IconSectionList class="size-7 m-0.5"/>
  </a>
</nav>

<span class="-my-4"></span>

{@render children()}

