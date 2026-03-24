import { API_URL } from '$lib/constants.ts';

import type { LayoutLoad } from './$types.ts';
import type { TestInstance, TestItem } from '$lib/index.ts';
import { error } from '@sveltejs/kit';


export const load: LayoutLoad = async ({ fetch, params }) => {
  const { test_id } = params;

  let test_instance: TestInstance | undefined = undefined;
  let test_items: TestItem[] | undefined = undefined;

  try {
    const [instanceResponse, itemsResponse] = await Promise.all([
      fetch(`${API_URL}/api/test_instances/${test_id}`),
      fetch(`${API_URL}/api/test_items/${test_id}/items`)
    ]);
  
    if (instanceResponse.ok) {
      const result = await instanceResponse.json();
      test_instance = {
        test_id,
        name: result.name,
        section_id: result.section_id,
        date: result.date,
        is_done_rendering: result.is_done_rendering
      };
    }
  
    if (itemsResponse.ok) {
      const result = await itemsResponse.json();
      test_items = result.items;
    }
  } catch (e) {
    console.log("Failed to fetch test instances and test items.\n" + String(e));
    throw error(500, "Failed to fetch test instances and test items.");
  }

  return {
    test_id,
    test_instance,
    test_items,
  };
};
