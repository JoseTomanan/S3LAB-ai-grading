<script lang="ts">
  const { data, children } = $props();
  import { Button } from '$lib/components/CustomButton/index.ts';
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
  <Button variant="floating" href="/sections">
    <IconBack class="size-8"/>
  </Button>
  <span class="*:leading-5 *:text-center">
    <h2>{sectionName}</h2>
  </span>
  <Button variant="floating" href="/instances">
    <IconSectionList class="size-7 m-0.5"/>
  </Button>
</nav>

<span class="-my-4"></span>

{@render children()}

